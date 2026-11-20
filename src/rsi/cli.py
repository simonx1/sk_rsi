"""The rsi command line. Exit codes: 0 success, 1 a run that stopped on an error, 2 invalid config or arguments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import config, loop, outcomes, promote, sim, tasks, workspace
from .state import Layout, LockError, RunState

TEMPLATE = """\
# How this project plugs into the RSI loop. Commit this file on the base branch.

[project]
name = "{name}"              # names the state directory (~/.rsi/{name}/) and nothing else
base_branch = "main"

[setting]
benchmark = "{name}"         # a label; the framework appends -v<n>
model = "claude-opus-5-5"    # a label for the model that executes the tasks

[harness]
editable = ["src/**"]        # what the loop may change
frozen = []                  # never changed, even when it matches editable; list your scorer here

[tasks]
# One JSON object per line, each with a unique "id". It must stay outside the repo.
# file = "~/.rsi/{name}/private/tasks.jsonl"    # this is the default
heldout_fraction = 0.5
rollouts = 1
pass_threshold = 0.5
heldout = "endpoints"        # or "every_round"

[commands]
# Placeholders arrive shell-quoted, so do not quote them: {task_file} {task_id} {checkout} {out} {rollout}
# run_task must write {out}/result.json: {"score": 0..1, "summary": "optional text"}
run_task = "bin/rsi-run-task {task_file} {checkout} {out}"
# check = "make test"
# ingest = "bin/rsi-outcomes --since {since}"

[budget]
rounds = 5
candidates = 3
parallel = 2
task_timeout_s = 3600
agent_timeout_s = 1800
max_wall_clock_h = 48
tolerance = 1.0              # score points a candidate may lose and still be accepted
max_infra_failure_rate = 0.3

[promotion]
pull_request = true
"""


def _rounds(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _project(args: argparse.Namespace) -> Path:
    return workspace.repo_root(Path(args.project or "."))


def _load(project: Path) -> tuple[config.Config, Layout]:
    cfg = config.load(project / loop.CONFIG_PATH)
    return cfg, Layout(cfg.name)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _promotion_lines(result: dict) -> list[str]:
    if "url" in result:
        lines = ["Pull request: " + result["url"] + (" (draft: the held-out score went down)" if result["draft"] else "")]
    elif "error" in result:
        lines = ["Pull request: failed: " + result["error"]]
    else:
        lines = ["Pull request: not opened: " + result["skipped"]]
    if result.get("manual"):
        lines.append("To promote by hand:")
        lines += [f"  {command}" for command in result["manual"]]
    return lines


def _status_lines(st: RunState, layout: Layout) -> list[str]:
    headline = f"Run {st.run} ({st.branch}): {st.status}"
    if st.error:
        headline += f" ({st.error})"
    lines = [
        headline,
        f"Setting: {st.benchmark}, {st.config['model']}",
        f"Round {min(st.round, st.rounds)} of {st.rounds}, next step: {st.step}",
    ]
    if st.scores:
        lines.append("Evolve: " + " → ".join(f"{score:.2f}" for score in st.scores))
    if "start" in st.heldout:
        rounds = st.heldout.get("rounds", {})
        if rounds:  # scored after every round; the last of these is the final score
            points = [st.heldout["start"]] + [rounds[key] for key in sorted(rounds, key=int)]
            lines.append("Held-out: " + " → ".join(f"{score:.2f}" for score in points))
        else:
            end = f"{st.heldout['end']:.2f}" if "end" in st.heldout else "not scored yet"
            lines.append(f"Held-out: {st.heldout['start']:.2f} → {end}")
    if st.promoted:
        lines += _promotion_lines(st.promoted)
    lines.append(f"Pending tasks from live outcomes: {tasks.pending_count(layout)}")
    return lines


def cmd_init(args: argparse.Namespace) -> int:
    project = _project(args)
    path = project / loop.CONFIG_PATH
    if path.exists():
        print(f"rsi: {path} already exists", file=sys.stderr)
        return 2
    name = re.sub(r"[^A-Za-z0-9._-]", "-", project.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.replace("{name}", name))
    layout = Layout(name)
    layout.private.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {path}")
    print(f"Put the benchmark's tasks in {layout.tasks_path(None)}, one JSON object per line.")
    print("Then edit the config, commit it, and run: rsi validate")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    project = _project(args)
    cfg, layout = _load(project)
    checked = loop.validate(project, cfg, layout, save_split=False)
    for warning in checked.warnings:
        print(f"warning: {warning}")
    for error in checked.errors:
        print(f"error: {error}")
    print(f"evolve tasks: {len(checked.evolve)}, held-out tasks: {len(checked.heldout)}")
    problem = loop.run_check(cfg, project)
    if not cfg.check:
        print("check: not configured")
    elif problem:
        print(f"error: {problem}")
    else:
        print("check: passed")
    return 2 if checked.errors or problem else 0


def _dry_run(project: Path, rounds: int | None) -> int:
    cfg, layout = _load(project)
    checked = loop.validate(project, cfg, layout, save_split=False)
    if checked.errors:
        for error in checked.errors:
            print(f"error: {error}")
        return 2
    rounds = cfg.rounds if rounds is None else rounds
    total = loop.planned_task_runs(len(checked.evolve), len(checked.heldout), cfg.rollouts, rounds,
                                   cfg.candidates, cfg.heldout == "every_round")
    print(f"up to {total} task runs")
    print(f"  {_plural(len(checked.evolve), 'evolve task')}, {_plural(len(checked.heldout), 'held-out task')}, "
          f"{_plural(cfg.rollouts, 'rollout')} each")
    print(f"  {_plural(rounds, 'round')}, up to {_plural(cfg.candidates, 'candidate')} evaluated per round")
    print(f"  held-out scored {'after every round' if cfg.heldout == 'every_round' else 'at the start and the end'}")
    print(f"  each task run may take up to {cfg.task_timeout_s}s; at most {cfg.parallel} run at once")
    return 0


def _run(args: argparse.Namespace, project: Path) -> int:
    st = loop.run(project, rounds=args.rounds, from_ref=args.from_ref,
                  promote=not args.no_promote, new=args.new)
    cfg, layout = _load(project)
    print("\n".join(_status_lines(st, layout)))
    return 1 if st.promoted and "error" in st.promoted else 0


def cmd_run(args: argparse.Namespace) -> int:
    project = _project(args)
    if args.dry_run:
        return _dry_run(project, args.rounds)
    return _run(args, project)


def cmd_continue(args: argparse.Namespace) -> int:
    project = _project(args)
    if args.dry_run:
        return _dry_run(project, args.rounds)
    cfg, layout = _load(project)
    unfinished = loop.find_unfinished(layout)
    if unfinished and not args.new:
        raise loop.RunError(f"run {unfinished.run} is unfinished; finish it with `rsi run` or pass --new to abandon it")
    added = tasks.fold_pending(layout, layout.tasks_path(cfg.tasks_file))
    print(f"added {_plural(added, 'task')}; the benchmark is now {tasks.benchmark_label(cfg.benchmark, layout)}")
    return _run(args, project)


def cmd_status(args: argparse.Namespace) -> int:
    project = _project(args)
    _, layout = _load(project)
    st = loop.latest_run(layout)
    if st is None:
        print("No runs yet.")
        return 0
    print("\n".join(_status_lines(st, layout)))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    project = _project(args)
    cfg, layout = _load(project)
    if not cfg.ingest:
        print("No ingest command is configured.")
        return 0
    got = outcomes.ingest(cfg, layout, project)
    print(f"{_plural(got['evidence'], 'evidence record')}, {_plural(got['tasks'], 'new task')} queued, "
          f"{_plural(len(got['skipped']), 'line')} skipped")
    for note in got["skipped"]:
        print(f"  {note}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    project = _project(args)
    _, layout = _load(project)
    if args.run:
        state_file = layout.run(args.run).state_file
        if not state_file.is_file():
            raise loop.RunError(f"no run named {args.run}")
        st = RunState.load(state_file)
    else:
        st = loop.latest_run(layout)
        if st is None:
            raise loop.RunError("there are no runs to promote")
    if st.status != "done":
        raise loop.RunError(f"run {st.run} is {st.status}; only a finished run can be promoted")
    if st.promoted and st.promoted.get("url"):  # already opened; asking again must not lose it
        print("\n".join(_promotion_lines(st.promoted)))
        return 0
    try:
        memory = json.loads(workspace.show(project, st.branch, loop.SIM_PATH))
    except workspace.GitError:
        memory = sim.empty()
    st.promoted = promote.promote(project, st, config.Config.from_dict(st.config), memory)
    st.save(layout.run(st.run).state_file)
    print("\n".join(_promotion_lines(st.promoted)))
    return 1 if "error" in st.promoted else 0


def cmd_sim_show(args: argparse.Namespace) -> int:
    print(sim.render_text(sim.load(_project(args) / loop.SIM_PATH)))
    return 0


def cmd_sim_merge(args: argparse.Namespace) -> int:
    path = _project(args) / loop.SIM_PATH
    memory = sim.load(path)
    sim.merge(memory, args.src, args.dst)
    sim.save(path, memory)
    print(f"merged {args.src} into {args.dst} in {path}; commit the file to keep the change")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", help="path inside the project's git repo (default: the current directory)")
    parser = argparse.ArgumentParser(prog="rsi", description="Recursive self-improvement loops for any project.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[common], help=text, description=text)
        sub.set_defaults(func=func)
        return sub

    add("init", cmd_init, "Write a commented .rsi/config.toml and create the state directory.")
    add("validate", cmd_validate, "Check the config, the task file and the project's check command.")
    for name, func, text in (
        ("run", cmd_run, "Resume the unfinished run, or start one."),
        ("continue", cmd_continue, "Fold pending live tasks into the benchmark, then run."),
    ):
        sub = add(name, func, text)
        sub.add_argument("--dry-run", action="store_true", help="print how many task runs this would start, then exit")
        sub.add_argument("--rounds", type=_rounds, help="override budget.rounds for this run")
        sub.add_argument("--from", dest="from_ref", metavar="BRANCH", help="start from this branch instead of the base branch")
        sub.add_argument("--no-promote", action="store_true", help="do not push or open a pull request")
        sub.add_argument("--new", action="store_true", help="abandon the unfinished run and start fresh")
    add("status", cmd_status, "Show the latest run.")
    add("ingest", cmd_ingest, "Run the ingest command now.")
    sub = add("promote", cmd_promote, "Push a finished run's branch and open its pull request.")
    sub.add_argument("run", nargs="?", help="run id (default: the latest run)")

    memory = commands.add_parser("sim", help="Read or tidy the Self-Improvement Memory.")
    memory_commands = memory.add_subparsers(dest="sim_command", required=True)
    show = memory_commands.add_parser("show", parents=[common], help="Print the memory as readable text.")
    show.set_defaults(func=cmd_sim_show)
    merge = memory_commands.add_parser("merge", parents=[common], help="Merge one problem entry into another.")
    merge.add_argument("src", metavar="FROM")
    merge.add_argument("dst", metavar="INTO")
    merge.set_defaults(func=cmd_sim_merge)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (config.ConfigError, tasks.TaskError) as error:
        print(f"rsi: {error}", file=sys.stderr)
        return 2
    except (loop.RunError, LockError, workspace.GitError, sim.SimError, outcomes.OutcomeError) as error:
        print(f"rsi: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("rsi: interrupted. A run in progress keeps its state; `rsi run` resumes it.", file=sys.stderr)
        return 130
