"""The loop: a state machine whose state is saved after every step.

    start -> heldout_start -> baseline
          -> per round: ingest -> propose -> screen -> evaluate -> select -> record
                        (-> heldout_round when tasks.heldout = "every_round")
          -> heldout_end -> promote -> done

Every step can be run again after a crash: finished rollouts are read back from
disk, and the steps that write to git or the memory check before they write.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import agents, config, evaluator, outcomes, prompts, shell, sim, tasks, workspace
from . import promote as promote_mod
from .state import Layout, Lock, RunLayout, RunState

CONFIG_PATH = ".rsi/config.toml"
SIM_PATH = ".rsi/sim.json"
LOG_PATH = ".rsi/log.md"


class RunError(Exception):
    pass


def _say(message: str) -> None:
    print(f"[rsi] {message}", file=sys.stderr, flush=True)


def planned_task_runs(evolve: int, heldout: int, rollouts: int, rounds: int, candidates: int, every_round: bool) -> int:
    """The most task runs a run can start: every candidate survives screening in every round."""
    evolve_runs = evolve * rollouts * (1 + rounds * candidates)
    heldout_runs = heldout * rollouts * ((rounds + 1) if every_round else 2)
    return evolve_runs + heldout_runs


# --- validation -------------------------------------------------------------

@dataclass
class Validation:
    errors: list[str]
    warnings: list[str]
    evolve: list[dict]
    heldout: list[dict]


def validate(project: Path, cfg: config.Config, layout: Layout) -> Validation:
    """Everything that can be checked without running a task."""
    errors: list[str] = []
    warnings: list[str] = []
    evolve: list[dict] = []
    heldout: list[dict] = []
    path = layout.tasks_path(cfg.tasks_file)
    if path.resolve().is_relative_to(project.resolve()):
        errors.append(f"the task file {path} is inside the repo, where agents can read the expected answers")
    try:
        all_tasks = tasks.load_tasks(path)
    except tasks.TaskError as error:
        errors.append(str(error))
        all_tasks = []
    if all_tasks:
        split = tasks.load_split(layout, [t["id"] for t in all_tasks], cfg.heldout_fraction)
        evolve = [t for t in all_tasks if split[t["id"]] == "evolve"]
        heldout = [t for t in all_tasks if split[t["id"]] == "heldout"]
        if not evolve or not heldout:
            errors.append(
                f"both sets need at least one task: there are {len(evolve)} evolve and {len(heldout)} held-out")
    for name in ("run_task", "check", "ingest"):
        command = getattr(cfg, name)
        if command and workspace.command_in_editable(command, project, cfg.editable, cfg.frozen):
            errors.append(f"commands.{name} runs a file the loop may edit; add it to harness.frozen")
    if cfg.proposer or cfg.selector:
        warnings.append(
            "a custom agent command is configured: held-out data is only kept from the default agents, "
            "which have no shell")
    return Validation(errors, warnings, evolve, heldout)


def denied_paths(cfg: config.Config, layout: Layout) -> list[Path]:
    """Everything an agent must not read: the private directory, and the task file when it is kept elsewhere."""
    denied = [layout.private]
    tasks_path = layout.tasks_path(cfg.tasks_file)
    if not tasks_path.resolve().is_relative_to(layout.private.resolve()):
        denied.append(tasks_path)
    return denied


def run_check(cfg: config.Config, cwd: Path) -> str | None:
    """Run the project's fast check. Returns None when it passes, otherwise what went wrong."""
    if not cfg.check:
        return None
    done = shell.run(cfg.check, cwd=cwd, timeout=cfg.task_timeout_s)
    if done.timed_out:
        return f"check timed out after {cfg.task_timeout_s}s"
    if done.returncode != 0:
        tail = (done.stdout + done.stderr).strip()[-400:]
        return f"check failed with exit code {done.returncode}" + (f": {tail}" if tail else "")
    return None


# --- context ----------------------------------------------------------------

@dataclass
class Ctx:
    project: Path
    cfg: config.Config
    layout: Layout
    rl: RunLayout
    state: RunState
    evolve: list[dict]
    heldout: list[dict]
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def main_wt(self) -> Path:
        return self.rl.worktrees / "main"

    def save(self) -> None:
        self.state.save(self.rl.state_file)

    def hist(self) -> Path:
        return self.rl.history / f"round-{self.state.round}"

    def agent_dir(self, name: str) -> Path:
        return self.rl.agents / f"round-{self.state.round}" / name


def _load_sim(worktree: Path) -> dict:
    try:
        return sim.load(worktree / SIM_PATH)
    except sim.SimError as error:
        raise RunError(str(error)) from None


def _runs(layout: Layout) -> list[RunState]:
    """Every run of the project, newest first."""
    if not layout.private_runs.is_dir():
        return []
    found = []
    for directory in sorted(layout.private_runs.iterdir(), reverse=True):
        state_file = directory / "state.json"
        if state_file.is_file():
            found.append(RunState.load(state_file))
    return found


def find_unfinished(layout: Layout) -> RunState | None:
    return next((st for st in _runs(layout) if st.status in ("running", "error")), None)


def latest_run(layout: Layout) -> RunState | None:
    return next((st for st in _runs(layout) if st.status != "abandoned"), None)


def _new_run_id(project: Path, layout: Layout, now: datetime) -> str:
    base = now.strftime("%Y%m%d-%H%M")
    run_id, n = base, 1
    while layout.run(run_id).private.exists() or workspace.branch_exists(project, f"rsi/{run_id}"):
        n += 1
        run_id = f"{base}-{n}"
    return run_id


# --- starting, resuming, abandoning ----------------------------------------

def _start(project: Path, live: config.Config, layout: Layout,
           from_ref: str | None, rounds: int | None, now: datetime) -> RunState:
    base_ref = from_ref or live.base_branch
    try:
        text = workspace.show(project, base_ref, CONFIG_PATH)
    except workspace.GitError:
        raise RunError(f"{CONFIG_PATH} is not committed on {base_ref}; commit it there first") from None
    try:
        cfg = config.parse(text)
    except config.ConfigError as error:
        raise RunError(f"{CONFIG_PATH} on {base_ref}: {error}") from None
    if cfg.name != live.name:
        raise RunError(f"project.name is {cfg.name!r} on {base_ref} but {live.name!r} in the working copy")
    checked = validate(project, cfg, layout)
    for warning in checked.warnings:
        _say(f"warning: {warning}")
    if checked.errors:
        raise RunError("; ".join(checked.errors))
    previous = latest_run(layout)
    if previous and previous.status == "done" and not workspace.is_merged(project, previous.branch, base_ref):
        _say(f"warning: the previous run's branch {previous.branch} is not merged into {base_ref}, so its "
             f"edits and memory are not part of this run. Use --from {previous.branch} to build on it.")

    run_id = _new_run_id(project, layout, now)
    rl = layout.run(run_id)
    branch = f"rsi/{run_id}"
    main_wt = rl.worktrees / "main"
    workspace.add_worktree(project, main_wt, branch, base_ref)
    problem = run_check(cfg, main_wt)
    if problem:
        workspace.remove_worktree(project, main_wt)
        workspace.delete_branch(project, branch)
        raise RunError(f"the starting harness does not pass check: {problem}")

    label = tasks.benchmark_label(cfg.benchmark, layout)
    memory = _load_sim(main_wt)
    last = sim.latest_attempt(memory)
    if last and last["domain"] != {"benchmark": label, "model": cfg.model}:
        sim.add_hop(memory, {
            "run": last["run"], "benchmark": last["domain"]["benchmark"],
            "model": last["domain"]["model"], "into": run_id,
        })
        sim.save(main_wt / SIM_PATH, memory)
        workspace.commit_all(main_wt, f"rsi: carry the memory into {run_id}", paths=[".rsi"])
    head = workspace.rev_parse(main_wt)
    st = RunState(
        run=run_id, project=str(project), branch=branch, base_ref=base_ref, config=cfg.to_dict(),
        benchmark=label, rounds=cfg.rounds if rounds is None else rounds,
        evolve_ids=[t["id"] for t in checked.evolve], heldout_ids=[t["id"] for t in checked.heldout],
        start_sha=head, current_sha=head, eval_sha=head,
    )
    st.save(rl.state_file)
    _say(f"started run {run_id} on {branch} ({label}, {cfg.model})")
    return st


def _context(project: Path, layout: Layout, st: RunState) -> Ctx:
    cfg = config.Config.from_dict(st.config)
    try:
        by_id = {t["id"]: t for t in tasks.load_tasks(layout.tasks_path(cfg.tasks_file))}
    except tasks.TaskError as error:
        raise RunError(str(error)) from None
    missing = [task_id for task_id in st.evolve_ids + st.heldout_ids if task_id not in by_id]
    if missing:
        raise RunError(f"{len(missing)} task(s) of this run are no longer in the task file")
    ctx = Ctx(project, cfg, layout, layout.run(st.run), st,
              [by_id[i] for i in st.evolve_ids], [by_id[i] for i in st.heldout_ids])
    if not ctx.main_wt.exists():
        workspace.add_worktree(project, ctx.main_wt, st.branch, st.branch)
    st.status, st.error = "running", None
    return ctx


def _abandon(project: Path, layout: Layout, st: RunState) -> None:
    rl = layout.run(st.run)
    if rl.worktrees.is_dir():
        for child in rl.worktrees.iterdir():
            workspace.remove_worktree(project, child)
    for candidate in st.candidates:
        if candidate.get("branch"):
            workspace.delete_branch(project, candidate["branch"])
    st.status = "abandoned"
    st.save(rl.state_file)
    _say(f"abandoned run {st.run}; its branch {st.branch} is kept")


def run(project: Path, rounds: int | None = None, from_ref: str | None = None,
        promote: bool = True, new: bool = False, now: datetime | None = None) -> RunState:
    """Resume the project's unfinished run, or start one, and drive it to the end."""
    project = workspace.repo_root(project)
    try:
        live = config.load(project / CONFIG_PATH)
    except config.ConfigError as error:
        raise RunError(str(error)) from None
    layout = Layout(live.name)
    with Lock(layout.lock_file):
        st = find_unfinished(layout)
        if st is not None and new:
            _abandon(project, layout, st)
            st = None
        if st is None:
            st = _start(project, live, layout, from_ref, rounds, now or datetime.now())
        else:
            _say(f"resuming run {st.run} at round {st.round}, step {st.step}")
        ctx = _context(project, layout, st)
        _drive(ctx, promote)
        return ctx.state


# --- the state machine ------------------------------------------------------

def _drive(ctx: Ctx, promote: bool) -> None:
    st = ctx.state
    while st.step != "done":
        began = time.monotonic()
        _say(f"{st.run} round {st.round}: {st.step}")
        try:
            if st.step == "promote" and not promote:
                st.promoted = {"skipped": "--no-promote"}
                st.step = "done"
            else:
                _HANDLERS[st.step](ctx)
        except RunError as error:
            st.status, st.error = "error", str(error)
            ctx.save()
            raise
        st.active_seconds += time.monotonic() - began
        ctx.save()
    st.status = "done"
    ctx.save()
    workspace.remove_worktree(ctx.project, ctx.main_wt)


def _evolve_now(ctx: Ctx) -> evaluator.EvalResult:
    """Evolve results of the current harness. Read back from disk once they exist."""
    st = ctx.state
    return evaluator.evaluate(ctx.cfg, ctx.evolve, ctx.main_wt, ctx.rl.feedback / st.eval_sha, st.benchmark)


def _heldout_now(ctx: Ctx) -> evaluator.EvalResult:
    st = ctx.state
    return evaluator.evaluate(ctx.cfg, ctx.heldout, ctx.main_wt, ctx.rl.heldout / st.eval_sha, st.benchmark)


def _stop_if_evaluator_is_broken(ctx: Ctx, result: evaluator.EvalResult, what: str) -> None:
    rate = result.infra_failure_rate()
    limit = ctx.cfg.max_infra_failure_rate
    if rate <= limit:
        return
    st = ctx.state
    # Forget the failed results so the next attempt runs them again, and start over from the first step.
    shutil.rmtree(ctx.rl.heldout / st.eval_sha, ignore_errors=True)
    shutil.rmtree(ctx.rl.feedback / st.eval_sha, ignore_errors=True)
    st.step = "heldout_start"
    raise RunError(
        f"{rate:.0%} of the {what} rollouts failed to run (the limit is {limit:.0%}). "
        "Fix run_task or its environment and run again, or start over with --new.")


def _out_of_time(ctx: Ctx) -> bool:
    return ctx.state.active_seconds > ctx.cfg.max_wall_clock_h * 3600


def _next_round(ctx: Ctx) -> None:
    st = ctx.state
    if st.round >= st.rounds:
        st.step = "heldout_end"
    elif _out_of_time(ctx):
        _say(f"the wall-clock budget of {ctx.cfg.max_wall_clock_h}h is used up; no further rounds")
        st.step = "heldout_end"
    else:
        st.step = "ingest"


def _heldout_start(ctx: Ctx) -> None:
    result = _heldout_now(ctx)
    _stop_if_evaluator_is_broken(ctx, result, "held-out")
    ctx.state.heldout["start"] = result.score()
    ctx.state.step = "baseline"


def _baseline(ctx: Ctx) -> None:
    result = _evolve_now(ctx)
    _stop_if_evaluator_is_broken(ctx, result, "evolve")
    ctx.state.scores = [result.score()]
    _say(f"starting evolve score: {result.score():.2f}")
    _next_round(ctx)


def _heldout_end(ctx: Ctx) -> None:
    ctx.state.heldout["end"] = _heldout_now(ctx).score()
    ctx.state.step = "promote"


def _promote(ctx: Ctx) -> None:
    st = ctx.state
    st.promoted = promote_mod.promote(ctx.project, st, ctx.cfg, _load_sim(ctx.main_wt))
    st.step = "done"


# --- one round --------------------------------------------------------------

LOG_HEADER = (
    "# RSI log\n\n"
    "Evolve-set scores, one row per round.\n\n"
    "| Run | Round | Before | After | Accepted | Problem | Fixed | Missed | Regressed |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fresh_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _first_line(text: str) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= 72 else line[:69] + "..."


def _drop(candidate: dict, reason: str) -> None:
    candidate["status"] = "dropped"
    candidate["reason"] = reason


def _with_status(ctx: Ctx, status: str) -> list[dict]:
    return [c for c in ctx.state.candidates if c["status"] == status]


def _call_agent(ctx: Ctx, role: str, prompt: str, cwd: Path, out: Path, env: dict[str, str]) -> agents.AgentResult:
    command = ctx.cfg.proposer if role == "proposer" else ctx.cfg.selector
    env = {"RSI_ROUND": str(ctx.state.round), **env}
    return agents.run_agent(role, command, prompt, cwd, out, ctx.rl.feedback, ctx.rl.history,
                            denied_paths(ctx.cfg, ctx.layout), ctx.cfg.agent_timeout_s, env)


def _write_history(ctx: Ctx) -> None:
    """Save every candidate of this round, accepted or not, where later rounds can read it."""
    st = ctx.state
    keys = ("sample", "status", "reason", "sha", "files", "score", "fixed", "missed", "regressed", "passed", "total")
    for c in st.candidates:
        folder = ctx.hist() / c["sample"]
        folder.mkdir(parents=True, exist_ok=True)
        if c.get("manifest"):
            (folder / "manifest.json").write_text(json.dumps(c["manifest"], indent=2) + "\n")
        if c.get("sha") and c.get("files"):
            (folder / "diff.patch").write_text(workspace.diff_text(ctx.project, st.current_sha, c["sha"]) + "\n")
        (folder / "result.json").write_text(json.dumps({k: c.get(k) for k in keys}, indent=2) + "\n")
    if st.decision is not None:
        (ctx.hist() / "decision.json").write_text(json.dumps(st.decision, indent=2) + "\n")


def _ingest(ctx: Ctx) -> None:
    st = ctx.state
    ctx.hist().mkdir(parents=True, exist_ok=True)
    evidence_file = ctx.hist() / "evidence.jsonl"
    if not evidence_file.exists():
        if ctx.cfg.ingest:
            try:
                got = outcomes.ingest(ctx.cfg, ctx.layout, ctx.project)
                _say(f"ingest: {got['evidence']} evidence, {got['tasks']} new tasks, "
                     f"{len(got['skipped'])} lines skipped")
            except outcomes.OutcomeError as error:
                _say(f"warning: {error}; the round goes on without new live outcomes")
        records, cursor = outcomes.new_evidence(ctx.layout, set(st.heldout_ids))
        evidence_file.write_text("".join(json.dumps(r) + "\n" for r in records))
        outcomes.mark_shown(ctx.layout, cursor)
    st.candidates = [
        {"sample": f"s{i}", "status": "new", "agent_done": False, "agent_error": None}
        for i in range(1, ctx.cfg.candidates + 1)
    ]
    st.decision = None
    st.step = "propose"


def _propose(ctx: Ctx) -> None:
    st, cfg = ctx.state, ctx.cfg
    memory = _load_sim(ctx.main_wt)
    table = _evolve_now(ctx).table(cfg.pass_threshold)
    evidence = _read_jsonl(ctx.hist() / "evidence.jsonl")

    # Worktrees are created one at a time: concurrent `git worktree add` calls can collide.
    waiting = [c for c in st.candidates if c["status"] == "new" and not c["agent_done"]]
    for c in waiting:
        name = f"r{st.round}-{c['sample']}"
        worktree, branch = ctx.rl.worktrees / name, f"rsi-cand/{st.run}/{name}"
        workspace.reset_worktree(ctx.project, worktree, branch, st.current_sha)
        c["worktree"], c["branch"] = str(worktree), branch
    ctx.save()

    def call(c: dict) -> None:
        out = _fresh_dir(ctx.agent_dir(c["sample"]))
        prompt = prompts.proposer_prompt(cfg, st.benchmark, memory, table, evidence, st.scores[-1],
                                         st.eval_sha, ctx.rl.feedback, ctx.rl.history, out)
        result = _call_agent(ctx, "proposer", prompt, Path(c["worktree"]), out, {"RSI_SAMPLE": c["sample"]})
        with ctx.lock:
            c["agent_done"], c["agent_error"] = True, result.error
            ctx.save()

    with ThreadPoolExecutor(max_workers=cfg.parallel) as pool:
        list(pool.map(call, waiting))

    for c in _with_status(ctx, "new"):
        if c["agent_error"]:
            _drop(c, f"agent failed: {c['agent_error']}")
            continue
        try:
            c["manifest"] = agents.read_manifest(ctx.agent_dir(c["sample"]))
        except agents.AgentOutputError as error:
            _drop(c, f"invalid manifest: {error}")
            continue
        worktree = Path(c["worktree"])
        workspace.commit_all(worktree, f"rsi: {st.run} r{st.round} {c['sample']}: {_first_line(c['manifest']['what'])}")
        c["sha"] = workspace.rev_parse(worktree)
        c["files"] = workspace.changed_files(ctx.project, st.current_sha, c["sha"])
        c["status"] = "proposed"
    st.step = "screen"


def _code_screen(ctx: Ctx, c: dict, memory: dict, failing: set[str]) -> str | None:
    """The checks that need no judgement. Returns why the candidate is dropped, or None."""
    manifest = c["manifest"]
    cause = manifest["root_cause"]
    if "id" in cause and sim.resolve(memory, cause["id"]) is None:
        return f"root cause {cause['id']} is not in the memory"
    wrong = [t for t in manifest["predicted"] if t not in failing]
    if wrong:
        return f"predicted tasks are not failing evolve tasks: {', '.join(wrong)}"
    if not c["files"]:
        return "the edit changed no files"
    violations = workspace.path_violations(c["files"], ctx.cfg.editable, ctx.cfg.frozen)
    if violations:
        return "edit outside the allowed paths: " + "; ".join(violations)
    return run_check(ctx.cfg, Path(c["worktree"]))


def _screen(ctx: Ctx) -> None:
    st, cfg = ctx.state, ctx.cfg
    memory = _load_sim(ctx.main_wt)
    failing = set(st.evolve_ids) - _evolve_now(ctx).passing(cfg.pass_threshold)
    for c in _with_status(ctx, "proposed"):
        reason = _code_screen(ctx, c, memory, failing)
        if reason:
            _drop(c, reason)
    survivors = _with_status(ctx, "proposed")
    if survivors:
        out = _fresh_dir(ctx.agent_dir("screen"))
        items = [
            {"sample": c["sample"], "manifest": c["manifest"], "files": c["files"],
             "diff": workspace.diff_text(ctx.project, st.current_sha, c["sha"])}
            for c in survivors
        ]
        prompt = prompts.screen_prompt(cfg, st.benchmark, memory, items, out)
        result = _call_agent(ctx, "selector", prompt, ctx.main_wt, out, {"RSI_STAGE": "screen"})
        verdicts = None
        if result.ok:
            try:
                verdicts = agents.read_screen(out, [c["sample"] for c in survivors])
            except agents.AgentOutputError as error:
                _say(f"warning: the selector's screen answer was unusable ({error}); keeping every candidate")
        else:
            _say(f"warning: the selector failed at screen ({result.error}); keeping every candidate")
        for c in survivors:
            keep, why = verdicts[c["sample"]] if verdicts else (True, "")
            if keep:
                c["status"] = "kept"
            else:
                _drop(c, f"screened out: {why}")
    _write_history(ctx)
    st.step = "evaluate"


def _evaluate(ctx: Ctx) -> None:
    st, threshold = ctx.state, ctx.cfg.pass_threshold
    kept = _with_status(ctx, "kept")
    targets = {c["sample"]: (Path(c["worktree"]), ctx.rl.feedback / c["sha"]) for c in kept}
    results = evaluator.evaluate_many(ctx.cfg, ctx.evolve, targets, st.benchmark) if targets else {}
    before = _evolve_now(ctx).passing(threshold)
    for c in kept:
        result = results[c["sample"]]
        fixed, missed, regressed = sim.outcome(c["manifest"]["predicted"], before, result.passing(threshold))
        passed, total = result.passed_total(threshold)
        c.update(status="evaluated", score=result.score(), fixed=fixed, missed=missed,
                 regressed=regressed, passed=passed, total=total)
    st.step = "select"


def _select(ctx: Ctx) -> None:
    st, cfg = ctx.state, ctx.cfg
    evaluated = _with_status(ctx, "evaluated")
    accept, reason = None, "no candidate reached evaluation"
    if evaluated:
        memory = _load_sim(ctx.main_wt)
        items = []
        for c in evaluated:
            result = evaluator.evaluate(cfg, ctx.evolve, Path(c["worktree"]), ctx.rl.feedback / c["sha"], st.benchmark)
            items.append({**{k: c[k] for k in ("sample", "manifest", "files", "score", "fixed", "missed", "regressed")},
                          "table": result.table(cfg.pass_threshold)})
        out = _fresh_dir(ctx.agent_dir("select"))
        prompt = prompts.select_prompt(cfg, st.benchmark, memory, st.scores[-1],
                                       _evolve_now(ctx).table(cfg.pass_threshold), items, out)
        result = _call_agent(ctx, "selector", prompt, ctx.main_wt, out, {"RSI_STAGE": "select"})
        if not result.ok:
            reason = f"selector failed: {result.error}"
        else:
            try:
                accept, reason = agents.read_decision(out, [c["sample"] for c in evaluated])
            except agents.AgentOutputError as error:
                accept, reason = None, f"selector failed: {error}"
        if accept:
            chosen = next(c for c in evaluated if c["sample"] == accept)
            if chosen["score"] < st.scores[-1] - cfg.tolerance:
                reason = (f"guard: {accept} scored {chosen['score']:.2f}, more than {cfg.tolerance:.2f} "
                          f"below the current {st.scores[-1]:.2f}; the selector's reason was: {reason}")
                accept = None
    st.decision = {"accept": accept, "reason": reason}
    _write_history(ctx)
    st.step = "record"


def _append_log(ctx: Ctx, chosen: dict | None, pid: str | None, before: float, after: float) -> None:
    st = ctx.state
    path = ctx.main_wt / LOG_PATH
    text = path.read_text() if path.is_file() else LOG_HEADER
    start = f"| {st.run} | {st.round} |"
    if start in text:
        return
    if chosen:
        accepted = f"{chosen['sample']}: {_first_line(chosen['manifest']['what'])}".replace("|", "/")
        cells = [accepted, pid or "", ", ".join(chosen["fixed"]), ", ".join(chosen["missed"]),
                 ", ".join(chosen["regressed"])]
    else:
        cells = ["none", "", "", "", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}{start} {before:.2f} | {after:.2f} | " + " | ".join(cells) + " |\n")


def _record(ctx: Ctx) -> None:
    """Keep the accepted edit, write it into the memory, and commit. Safe to run twice."""
    st, cfg = ctx.state, ctx.cfg
    accept = st.decision["accept"]
    before = after = st.scores[-1]
    chosen, pid, opened = None, None, False
    if accept:
        chosen = next(c for c in st.candidates if c["sample"] == accept)
        after = chosen["score"]
        workspace.fast_forward(ctx.main_wt, chosen["sha"])
        memory = _load_sim(ctx.main_wt)
        pid = sim.find_attempt(memory, st.run, st.round)
        if pid is None:
            pid = sim.record_attempt(memory, chosen["manifest"]["root_cause"], {
                "run": st.run, "round": st.round, "sample": chosen["sample"],
                "accepted": True, "backfilled": False,
                "what": chosen["manifest"]["what"], "files": chosen["files"],
                "predicted": chosen["manifest"]["predicted"], "fixed": chosen["fixed"],
                "missed": chosen["missed"], "regressed": chosen["regressed"],
                "passed": chosen["passed"], "total": chosen["total"],
                "score_before": before, "score_after": after,
                "domain": {"benchmark": st.benchmark, "model": cfg.model},
            })
            sim.save(ctx.main_wt / SIM_PATH, memory)
        problem = sim.problem(memory, pid)
        opened = (problem["opened_run"], problem["opened_round"]) == (st.run, st.round)
    _append_log(ctx, chosen, pid, before, after)
    workspace.commit_all(ctx.main_wt, f"rsi: {st.run} round {st.round} record", paths=[".rsi"])
    _write_history(ctx)

    st.rounds_log.append({
        "round": st.round, "accepted": accept, "problem": pid, "opened": opened,
        "what": chosen["manifest"]["what"] if chosen else None,
        "score_before": before, "score_after": after,
        "fixed": chosen["fixed"] if chosen else [], "missed": chosen["missed"] if chosen else [],
        "regressed": chosen["regressed"] if chosen else [],
        "reason": st.decision["reason"],
        "dropped": [{"sample": c["sample"], "reason": c["reason"]} for c in _with_status(ctx, "dropped")],
    })
    for c in st.candidates:
        if c.get("worktree"):
            workspace.remove_worktree(ctx.project, Path(c["worktree"]))
            workspace.delete_branch(ctx.project, c["branch"])
    if chosen:
        st.eval_sha = chosen["sha"]
    st.current_sha = workspace.rev_parse(ctx.main_wt)
    st.scores.append(after)
    st.candidates, st.decision = [], None
    _say(f"round {st.round}: " + (f"accepted {accept}, evolve {before:.2f} -> {after:.2f}" if accept
                                 else "nothing accepted"))
    st.round += 1
    if cfg.heldout == "every_round":
        st.step = "heldout_round"
    else:
        _next_round(ctx)


def _heldout_round(ctx: Ctx) -> None:
    st = ctx.state
    st.heldout.setdefault("rounds", {})[str(st.round - 1)] = _heldout_now(ctx).score()
    _next_round(ctx)


_HANDLERS: dict[str, Callable[[Ctx], None]] = {
    "heldout_start": _heldout_start,
    "baseline": _baseline,
    "ingest": _ingest,
    "propose": _propose,
    "screen": _screen,
    "evaluate": _evaluate,
    "select": _select,
    "record": _record,
    "heldout_round": _heldout_round,
    "heldout_end": _heldout_end,
    "promote": _promote,
}
