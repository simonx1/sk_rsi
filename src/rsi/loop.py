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

from . import agents, config, evaluator, measure, outcomes, prompts, shell, sim, tasks, workspace
from . import project as projects
from . import promote as promote_mod
from .state import Layout, Lock, RunLayout, RunState, read_json

CONFIG_PATH = projects.CONFIG_PATH
SIM_PATH = projects.SIM_PATH
LOG_PATH = projects.LOG_PATH


class RunError(Exception):
    pass


def _say(message: str) -> None:
    print(f"[rsi] {message}", file=sys.stderr, flush=True)


def planned_task_runs(evolve: int, heldout: int, rollouts: int, rounds: int, candidates: int,
                      every_round: bool, paired: bool = False, confirm: bool = False) -> int:
    """The most task runs a run can start: every candidate survives screening in every round.

    With paired measurement the current harness is scored again beside the candidates of every
    round, and held-out is scored once at the end, for the starting and the final harness.
    """
    evolve_runs = evolve * rollouts * (1 + rounds * (candidates + (1 if paired else 0) + (2 if confirm else 0)))
    heldout_runs = heldout * rollouts * ((rounds + 1) if every_round else 2)
    return evolve_runs + heldout_runs


# --- validation -------------------------------------------------------------

@dataclass
class Validation:
    errors: list[str]
    warnings: list[str]
    evolve: list[dict]
    heldout: list[dict]


def validate(project: Path, cfg: config.Config, layout: Layout, save_split: bool = True) -> Validation:
    """Everything that can be checked without running a task.

    Starting a run saves the side each new task is given, for good. `save_split=False` only
    reports what the sides would be, so checking a setup does not commit to a split.
    """
    errors: list[str] = []
    warnings: list[str] = []
    evolve: list[dict] = []
    heldout: list[dict] = []
    path = layout.tasks_path(cfg.tasks_file)
    if path.resolve().is_relative_to(project.resolve()):
        errors.append(f"the task file {path} is inside the repo, where agents can read the expected answers")
    loaded = True
    try:
        all_tasks = tasks.load_tasks(path)
    except tasks.TaskError as error:
        errors.append(str(error))
        all_tasks, loaded = [], False
    if loaded:
        ids = [t["id"] for t in all_tasks]
        if save_split:
            split = tasks.load_split(layout, ids, cfg.heldout_fraction)
        else:
            split = tasks.assign_split(ids, read_json(layout.split_file, {}), cfg.heldout_fraction)
        evolve = [t for t in all_tasks if split[t["id"]] == "evolve"]
        heldout = [t for t in all_tasks if split[t["id"]] == "heldout"]
        if not evolve or not heldout:
            errors.append(
                f"both sets need at least one task: there are {len(evolve)} evolve and {len(heldout)} held-out")
    for denied in denied_paths(cfg, layout):
        if any(ch in str(denied) for ch in "*?[]{}"):
            errors.append(
                f"{denied} contains pattern characters (* ? [ ] {{ }}), so agents cannot be denied it reliably; "
                "use a path without them")
    for name in ("run_task", "check", "ingest"):
        command = getattr(cfg, name)
        if command and workspace.command_in_editable(command, project, cfg.editable, cfg.frozen):
            errors.append(f"commands.{name} names a file the loop may edit; add it to harness.frozen, "
                          "or call it from a frozen script")
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
    step_began: float = 0.0   # time.monotonic() when the running step started

    @property
    def main_wt(self) -> Path:
        return self.rl.worktrees / "main"

    def save(self) -> None:
        self.state.save(self.rl.state_file)

    @property
    def sim_path(self) -> Path:
        return self.layout.sim_file if self.state.external else self.main_wt / SIM_PATH

    @property
    def log_path(self) -> Path:
        return self.layout.log_file if self.state.external else self.main_wt / LOG_PATH

    def hist(self) -> Path:
        return self.rl.history / f"round-{self.state.round}"

    def agent_dir(self, name: str) -> Path:
        return self.rl.agents / f"round-{self.state.round}" / name


def _read_sim(path: Path) -> dict:
    try:
        return sim.load(path)
    except sim.SimError as error:
        raise RunError(str(error)) from None


def _load_sim(ctx: Ctx) -> dict:
    return _read_sim(ctx.sim_path)


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

def _start(proj: projects.Project, from_ref: str | None, rounds: int | None, now: datetime) -> RunState:
    project, live, layout = proj.repo, proj.cfg, proj.layout
    base_ref = from_ref or live.base_branch
    if proj.external:
        cfg = live  # read from disk; nothing of the loop is committed in the repository
    else:
        try:
            text = workspace.show(project, base_ref, CONFIG_PATH)
        except workspace.GitError:
            raise RunError(f"{CONFIG_PATH} is not committed on {base_ref}; commit it there first") from None
        try:
            cfg = config.parse(text)
        except config.ConfigError as error:
            raise config.ConfigError(f"{CONFIG_PATH} on {base_ref}: {error}") from None
        if cfg.name != live.name:
            raise RunError(f"project.name is {cfg.name!r} on {base_ref} but {live.name!r} in the working copy")
    checked = validate(project, cfg, layout)
    for warning in checked.warnings:
        _say(f"warning: {warning}")
    if checked.errors:
        raise RunError("; ".join(checked.errors))
    previous = latest_run(layout)
    # Only for an in-repo project: there the memory travels on the run branch. An external project's
    # memory is one file, and its edits reach the base branch as separate pull requests.
    if (not proj.external and previous and previous.status == "done"
            and not workspace.is_merged(project, previous.branch, base_ref)):
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
    sim_path = layout.sim_file if proj.external else main_wt / SIM_PATH
    memory = _read_sim(sim_path)
    last = sim.latest_attempt(memory)
    if last and last["domain"] != {"benchmark": label, "model": cfg.model}:
        sim.add_hop(memory, {
            "run": last["run"], "benchmark": last["domain"]["benchmark"],
            "model": last["domain"]["model"], "into": run_id,
        })
        sim.save(sim_path, memory)
        if not proj.external:
            workspace.commit_all(main_wt, f"rsi: carry the memory into {run_id}", paths=[".rsi"])
    head = workspace.rev_parse(main_wt)
    st = RunState(
        run=run_id, project=str(project), branch=branch, base_ref=base_ref, config=cfg.to_dict(),
        benchmark=label, rounds=cfg.rounds if rounds is None else rounds,
        evolve_ids=[t["id"] for t in checked.evolve], heldout_ids=[t["id"] for t in checked.heldout],
        start_sha=head, current_sha=head, eval_sha=head, external=proj.external,
        step="baseline" if cfg.paired else "heldout_start",  # paired: held-out is scored once, at the end
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
    agents.set_writable(_shared(ctx), True)  # a run killed while agents ran left them read-only
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
    proj = projects.find(project)
    project, layout = proj.repo, proj.layout
    with Lock(layout.lock_file):
        st = find_unfinished(layout)
        if st is not None and new:
            _abandon(project, layout, st)
            st = None
        if st is None:
            st = _start(proj, from_ref, rounds, now or datetime.now())
        else:
            _say(f"resuming run {st.run} at round {st.round}, step {st.step}")
        ctx = _context(project, layout, st)
        _drive(ctx, promote)
        return ctx.state


# --- the state machine ------------------------------------------------------

def _drive(ctx: Ctx, promote: bool) -> None:
    st = ctx.state
    while st.step != "done":
        began = ctx.step_began = time.monotonic()
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
    return evaluator.evaluate(ctx.cfg, ctx.evolve, ctx.main_wt, ctx.rl.feedback / _eval_dir(st), st.benchmark,
                              current=True)


def _eval_dir(st: RunState) -> str:
    """The folder under feedback/ that holds the current harness's newest results."""
    return st.eval_dir or st.eval_sha


def _heldout_now(ctx: Ctx) -> evaluator.EvalResult:
    st = ctx.state
    return evaluator.evaluate(ctx.cfg, ctx.heldout, ctx.main_wt, ctx.rl.heldout / st.eval_sha, st.benchmark,
                              current=True)


def _stop_if_evaluator_is_broken(ctx: Ctx, result: evaluator.EvalResult, what: str) -> None:
    rate = result.infra_failure_rate()
    limit = ctx.cfg.max_infra_failure_rate
    if rate <= limit:
        return
    st = ctx.state
    # Forget the failed results so the next attempt runs them again, and start over from the first step.
    shutil.rmtree(ctx.rl.heldout / st.eval_sha, ignore_errors=True)
    shutil.rmtree(ctx.rl.feedback / _eval_dir(st), ignore_errors=True)
    st.step = "baseline" if ctx.cfg.paired else "heldout_start"
    raise RunError(
        f"{rate:.0%} of the {what} rollouts failed to run (the limit is {limit:.0%}). "
        "Fix run_task or its environment and run again, or start over with --new.")


def _out_of_time(ctx: Ctx) -> bool:
    """True when the active time, counting the step now running, is past the budget."""
    spent = ctx.state.active_seconds + (time.monotonic() - ctx.step_began)
    return spent > ctx.cfg.max_wall_clock_h * 3600


def _below_tolerance(current: float, candidate: float, tolerance: float) -> bool:
    """True when the candidate loses more than `tolerance` points. Scores have two decimals."""
    return round(current - candidate, 2) > tolerance


def _next_round(ctx: Ctx) -> None:
    st = ctx.state
    if st.round >= st.rounds:
        st.step = "heldout_end"
    elif _out_of_time(ctx):
        _say(f"the wall-clock budget of {ctx.cfg.max_wall_clock_h}h is used up; no further rounds")
        st.step = "heldout_end"
    else:
        st.step = "ingest"


def _failed(result: evaluator.EvalResult) -> int:
    return sum(1 for r in result.rollouts if r.error)


def _heldout_start(ctx: Ctx) -> None:
    result = _heldout_now(ctx)
    _stop_if_evaluator_is_broken(ctx, result, "held-out")
    ctx.state.heldout["start"] = result.score()
    ctx.state.heldout["failed"] = {"start": _failed(result), "final": 0, "total": len(result.rollouts)}
    ctx.state.step = "baseline"


def _baseline(ctx: Ctx) -> None:
    result = _evolve_now(ctx)
    _stop_if_evaluator_is_broken(ctx, result, "evolve")
    ctx.state.scores = [result.score()]
    _say(f"starting evolve score: {result.score():.2f}")
    _next_round(ctx)


def _heldout_end(ctx: Ctx) -> None:
    st = ctx.state
    if ctx.cfg.paired:
        _heldout_paired(ctx)
    else:
        result = _heldout_now(ctx)
        st.heldout["end"] = result.score()
        failed = st.heldout.setdefault("failed", {"start": 0, "final": 0, "total": len(result.rollouts)})
        failed["final"] = _failed(result)
    if any(st.heldout.get("failed", {}).get(side) for side in ("start", "final")):
        _say("warning: some held-out runs failed; the held-out numbers are not reliable")
    st.step = "promote"


def _heldout_paired(ctx: Ctx) -> None:
    """Score the starting and the final harness on the held-out tasks in one batch.

    Measured at the same time, the two numbers differ by what the run changed and not by what the
    machines did between its start and its end.
    """
    st = ctx.state
    targets = {"final": (ctx.main_wt, ctx.rl.heldout / "end-final")}
    start_wt = ctx.rl.worktrees / "start"
    if any(entry["accepted"] for entry in st.rounds_log):
        workspace.add_detached_worktree(ctx.project, start_wt, st.start_sha)
        targets = {"start": (start_wt, ctx.rl.heldout / "end-start"), **targets}
    results = evaluator.evaluate_many(ctx.cfg, ctx.heldout, targets, st.benchmark, current=set(targets))
    workspace.remove_worktree(ctx.project, start_wt)
    final = results["final"]
    start = results.get("start", final)
    st.heldout = {"start": start.score(), "end": final.score(),
                  "metrics_start": start.metric_means(), "metrics_end": final.metric_means(),
                  "failed": {"start": _failed(start), "final": _failed(final), "total": len(final.rollouts)}}
    # The limits once more, on this independent measurement of start against final: the run's final total.
    st.heldout["limits"] = measure.limit_status(ctx.cfg.constraints, {}, st.heldout["metrics_start"],
                                                st.heldout["metrics_end"])


def _promote(ctx: Ctx) -> None:
    st = ctx.state
    st.promoted = promote_mod.promote(ctx.project, st, ctx.cfg, _load_sim(ctx),
                                      scratch=ctx.rl.worktrees / "promote")
    st.step = "done"


# --- one round --------------------------------------------------------------

BASELINE = "@current"   # the current harness, when it is scored in a round's batch

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
    cfg = ctx.cfg
    command = (cfg.proposer if role == "proposer" else cfg.selector) or agents.builtin_command(
        cfg.agent_cli, cfg.agent_model, cfg.agent_effort, cfg.agent_timeout_s)
    env = {"RSI_ROUND": str(ctx.state.round), **env}
    # Agents read the shared folders; only the framework writes them. The selector also only reads its worktree.
    readonly = [ctx.rl.feedback, ctx.rl.history] + ([cwd] if role == "selector" else [])
    gitfile = workspace.read_gitfile(cwd)
    result = agents.run_agent(role, command, prompt, cwd, out, ctx.rl.feedback, ctx.rl.history,
                              denied_paths(ctx.cfg, ctx.layout), ctx.cfg.agent_timeout_s, env, readonly)
    if gitfile is not None and workspace.restore_gitfile(cwd, gitfile):
        # Put back before any git runs there: a replaced .git can point git at a config the agent wrote.
        _say(f"warning: the {role} changed {cwd}/.git; put back and its answer not used")
        result = agents.AgentResult(False, "the agent changed the worktree's .git")
    if role == "selector":
        # A selector that can write (a custom agent command) must still leave the run branch as it was.
        discarded = workspace.clean(cwd)
        if discarded:
            _say(f"warning: the selector changed {len(discarded)} path(s) in the run branch's worktree; discarded")
    return result


def _agent(ctx: Ctx, role: str, prompt: str, cwd: Path, out: Path, env: dict[str, str],
           read: Callable[[Path], object], reset: Callable[[], object] | None = None) -> tuple[object, tuple[str, str] | None]:
    """Call an agent and read its output file; when that fails, try once more.

    Returns (value, None), or (None, (kind, message)) with kind "agent" (the call failed) or "output"
    (it left no valid output). A timeout is not tried again. Before the second try `reset` is called,
    and the first try's folder is kept beside the new one as `<name>.try1`.
    """
    failure: tuple[str, str] | None = None
    for attempt in (1, 2):
        if attempt == 2:
            if reset is not None:
                reset()
            kept = out.with_name(out.name + ".try1")
            shutil.rmtree(kept, ignore_errors=True)
            out.rename(kept)
        _fresh_dir(out)
        result = _call_agent(ctx, role, prompt, cwd, out, env)
        if result.ok or result.denied:
            # An agent denied an action may have finished its work first; what it wrote decides.
            try:
                return read(out), None
            except agents.AgentOutputError as problem:
                failure = ("agent", result.error) if result.denied else ("output", str(problem))
        else:
            failure = ("agent", result.error)
            if result.timed_out:
                break
    return None, failure


def _shared(ctx: Ctx) -> list[Path]:
    """The folders agents read and only the framework writes."""
    return [ctx.rl.feedback, ctx.rl.history]


def _write_history(ctx: Ctx) -> None:
    """Save every candidate of this round, accepted or not, where later rounds can read it."""
    st = ctx.state
    keys = ("sample", "status", "reason", "sha", "files", "score", "fixed", "missed", "regressed",
            "passed", "total", "table", "metrics", "observed", "limits")
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
        # Only tasks the saved split puts on the evolve side: a queued or removed task may be held-out.
        split = read_json(ctx.layout.split_file, {})
        evolve_ids = {task_id for task_id, side in split.items() if side == "evolve"}
        records, cursor = outcomes.new_evidence(ctx.layout, evolve_ids)
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
    memory = _load_sim(ctx)
    current = _evolve_now(ctx)
    table = current.table(cfg.pass_threshold)
    metrics = current.metric_means()
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
        out = ctx.agent_dir(c["sample"])
        worktree = Path(c["worktree"])
        prompt = prompts.proposer_prompt(cfg, st.benchmark, memory, table, evidence, st.scores[-1],
                                         _eval_dir(st), ctx.rl.feedback, ctx.rl.history, out,
                                         metrics=metrics, limits=cfg.constraints, used=st.metric_ratio)
        manifest, failure = _agent(ctx, "proposer", prompt, worktree, out, {"RSI_SAMPLE": c["sample"]},
                                   agents.read_manifest, reset=lambda: workspace.clean(worktree))
        with ctx.lock:
            c["agent_done"] = True
            if failure:
                kind, message = failure
                c["agent_error"] = f"agent failed: {message}" if kind == "agent" else f"invalid manifest: {message}"
            else:
                c["manifest"] = manifest
            ctx.save()

    with agents.read_only(_shared(ctx)), ThreadPoolExecutor(max_workers=cfg.parallel) as pool:
        try:
            list(pool.map(call, waiting))
        except (KeyboardInterrupt, SystemExit):
            shell.kill_all()  # stop the agents now; their calls are made again on resume
            raise

    for c in _with_status(ctx, "new"):
        if c["agent_error"]:
            _drop(c, c["agent_error"])
            continue
        if "manifest" not in c:  # a run started before manifests were kept in the state
            try:
                c["manifest"] = agents.read_manifest(ctx.agent_dir(c["sample"]))
            except agents.AgentOutputError as error:
                _drop(c, f"invalid manifest: {error}")
                continue
        worktree = Path(c["worktree"])
        workspace.commit_all(worktree, f"rsi: {st.run} r{st.round} {c['sample']}: {_first_line(c['manifest']['what'])}")
        c["sha"] = workspace.rev_parse(worktree)
        c["files"] = workspace.changed_files(ctx.project, st.current_sha, c["sha"])
        workspace.clean(worktree)  # an ignored file is not in the commit, so it must not be there to be scored
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
    memory = _load_sim(ctx)
    failing = set(st.evolve_ids) - _evolve_now(ctx).passing(cfg.pass_threshold)
    for c in _with_status(ctx, "proposed"):
        reason = _code_screen(ctx, c, memory, failing)
        if reason:
            _drop(c, reason)
    survivors = _with_status(ctx, "proposed")
    if survivors:
        out = ctx.agent_dir("screen")
        items = [
            {"sample": c["sample"], "manifest": c["manifest"], "files": c["files"],
             "diff": workspace.diff_text(ctx.project, st.current_sha, c["sha"])}
            for c in survivors
        ]
        prompt = prompts.screen_prompt(cfg, st.benchmark, memory, items, out)
        samples = [c["sample"] for c in survivors]
        with agents.read_only(_shared(ctx)):
            verdicts, failure = _agent(ctx, "selector", prompt, ctx.main_wt, out, {"RSI_STAGE": "screen"},
                                       lambda folder: agents.read_screen(folder, samples))
        if failure:
            _say(f"warning: the selector failed at screen ({failure[1]}); keeping every candidate")
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
    for c in kept:
        workspace.clean(Path(c["worktree"]))  # the check may have left files; score exactly the commit
    targets = {c["sample"]: (Path(c["worktree"]), ctx.rl.feedback / c["sha"]) for c in kept}
    if ctx.cfg.paired and kept:
        # The current harness is measured again in this batch, so that it and the candidates meet
        # the same conditions. Its fresh result is the round's baseline.
        fresh = f"{st.eval_sha}.r{st.round}"
        targets = {BASELINE: (ctx.main_wt, ctx.rl.feedback / fresh), **targets}
    results = evaluator.evaluate_many(ctx.cfg, ctx.evolve, targets, st.benchmark, current={BASELINE}) if targets else {}
    if BASELINE in results:
        if any(r.error for r in results[BASELINE].rollouts):
            st.decision = {"accept": None, "baseline_failed": True,
                           "reason": "the round's baseline had failed runs; nothing can be compared"}
        else:
            st.eval_dir = fresh
    baseline = _evolve_now(ctx)
    for c in kept:
        _assess(ctx, c, results[c["sample"]], baseline)
        c["status"] = "evaluated"
    st.step = "select"


def _select(ctx: Ctx) -> None:
    st, cfg = ctx.state, ctx.cfg
    if st.decision and st.decision.get("baseline_failed"):
        _write_history(ctx)
        st.step = "record"
        return
    evaluated = _with_status(ctx, "evaluated")
    accept, reason = None, "no candidate reached evaluation"
    if evaluated:
        memory = _load_sim(ctx)
        baseline = _evolve_now(ctx)   # with paired measurement: scored in the same batch as the candidates
        current = baseline.score()
        items = []
        for c in evaluated:
            result = evaluator.evaluate(cfg, ctx.evolve, Path(c["worktree"]), ctx.rl.feedback / c["sha"], st.benchmark)
            keys = ("sample", "manifest", "files", "score", "fixed", "missed", "regressed", "metrics", "observed", "limits")
            items.append({**{k: c.get(k) for k in keys}, "table": result.table(cfg.pass_threshold)})
        out = ctx.agent_dir("select")
        prompt = prompts.select_prompt(cfg, st.benchmark, memory, current,
                                       baseline.table(cfg.pass_threshold), items, out)
        samples = [c["sample"] for c in evaluated]
        with agents.read_only(_shared(ctx)):
            decision, failure = _agent(ctx, "selector", prompt, ctx.main_wt, out, {"RSI_STAGE": "select"},
                                       lambda folder: agents.read_decision(folder, samples))
        if failure:
            reason = f"selector failed: {failure[1]}"
        else:
            accept, reason = decision
        if accept:
            # Rules in code that override the selector, in this order: limits, minimum gain, tolerance.
            chosen = next(c for c in evaluated if c["sample"] == accept)
            blocked = measure.blocking(chosen.get("limits") or [])
            gained = round(chosen["score"] - current, 2)
            said = f"; the selector's reason was: {reason}"
            if blocked:
                reason = f"guard: {accept} is past a limit: {measure.describe(blocked[0])}{said}"
                accept = None
            elif cfg.min_gain is not None and gained < cfg.min_gain:
                reason = f"guard: {accept} gained {gained:.2f} points, less than the {cfg.min_gain:.2f} required{said}"
                accept = None
            elif _below_tolerance(current, chosen["score"], cfg.tolerance):
                reason = (f"guard: {accept} scored {chosen['score']:.2f}, more than {cfg.tolerance:.2f} "
                          f"below the current {current:.2f}; the selector's reason was: {reason}")
                accept = None
    st.decision = {"accept": accept, "reason": reason}
    _write_history(ctx)
    st.step = "confirm" if accept and cfg.confirm else "record"


def _assess(ctx: Ctx, c: dict, result: evaluator.EvalResult, baseline: evaluator.EvalResult) -> None:
    """Fill a candidate's measured fields from its result, compared with the given baseline."""
    threshold = ctx.cfg.pass_threshold
    fixed, missed, regressed = sim.outcome(c["manifest"]["predicted"], baseline.passing(threshold),
                                           result.passing(threshold))
    passed, total = result.passed_total(threshold)
    base_metrics, metrics = baseline.metric_means(), result.metric_means()
    c.update(score=result.score(), fixed=fixed, missed=missed, regressed=regressed, passed=passed, total=total,
             table=result.table(threshold), metrics=metrics, observed=measure.changes(base_metrics, metrics),
             limits=measure.limit_status(ctx.cfg.constraints, ctx.state.metric_ratio, base_metrics, metrics))


def _confirm(ctx: Ctx) -> None:
    """Measure the chosen candidate once more beside the current version, and keep it only if it holds up.

    The candidate was chosen as the best of several noisy measurements, so its first numbers flatter
    it. This second, independent measurement decides, and its numbers are the ones recorded.
    """
    st, cfg = ctx.state, ctx.cfg
    chosen = next(c for c in st.candidates if c["sample"] == st.decision["accept"])
    workspace.clean(Path(chosen["worktree"]))
    base_dir, cand_dir = f"{st.eval_sha}.c{st.round}", f"{chosen['sha']}.c{st.round}"
    targets = {BASELINE: (ctx.main_wt, ctx.rl.feedback / base_dir), chosen["sample"]: (Path(chosen["worktree"]),
                                                                                       ctx.rl.feedback / cand_dir)}
    results = evaluator.evaluate_many(cfg, ctx.evolve, targets, st.benchmark, current={BASELINE})
    baseline, result = results[BASELINE], results[chosen["sample"]]
    again = "did not hold up when measured again"
    if any(r.error for r in baseline.rollouts):
        st.decision = {"accept": None, "reason": f"guard: {chosen['sample']} was not measured again: the "
                                                 "baseline had failed runs; nothing can be compared"}
    else:
        st.eval_dir = base_dir
        _assess(ctx, chosen, result, baseline)
        chosen["eval_dir"] = cand_dir
        current, gained = baseline.score(), round(chosen["score"] - baseline.score(), 2)
        blocked = measure.blocking(chosen["limits"])
        reason = None
        if blocked:
            reason = f"guard: {chosen['sample']} {again}: it is past a limit: {measure.describe(blocked[0])}"
        elif cfg.min_gain is not None and gained < cfg.min_gain:
            reason = (f"guard: {chosen['sample']} {again}: it gained {gained:.2f} points, "
                      f"less than the {cfg.min_gain:.2f} required")
        elif _below_tolerance(current, chosen["score"], cfg.tolerance):
            reason = (f"guard: {chosen['sample']} {again}: it scored {chosen['score']:.2f}, more than "
                      f"{cfg.tolerance:.2f} below the current {current:.2f}")
        if reason:
            st.decision = {"accept": None, "reason": f"{reason}; the selector's reason was: {st.decision['reason']}"}
        else:
            st.decision["confirmed"] = True
    _write_history(ctx)
    st.step = "record"


def _append_log(ctx: Ctx, chosen: dict | None, pid: str | None, before: float, after: float) -> None:
    st = ctx.state
    path = ctx.log_path
    text = path.read_text() if path.is_file() else LOG_HEADER
    start = f"| {st.run} | {st.round} |"
    if start in text:
        return
    if chosen:
        accepted = f"{chosen['sample']}: {_first_line(chosen['manifest']['what'])}".replace("|", "/")
        if chosen.get("observed"):
            accepted += " (" + ", ".join(f"{name} {change:+.2f}%" for name, change in chosen["observed"].items()) + ")"
        cells = [accepted, pid or "", ", ".join(chosen["fixed"]), ", ".join(chosen["missed"]),
                 ", ".join(chosen["regressed"])]
    else:
        cells = ["none", "", "", "", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}{start} {before:.2f} | {after:.2f} | " + " | ".join(cells) + " |\n")


def _measured_fields(chosen: dict, metrics_before: dict) -> dict:
    """What an attempt in the memory says about measurements. Empty for a project that reports none."""
    fields = {}
    if chosen["manifest"].get("expected"):
        fields["expected"] = chosen["manifest"]["expected"]
    if chosen.get("metrics"):
        fields.update(observed=chosen.get("observed", {}), metrics_before=metrics_before,
                      metrics_after=chosen["metrics"])
    return fields


def _record(ctx: Ctx) -> None:
    """Keep the accepted edit, write it into the memory, and commit. Safe to run twice."""
    st, cfg = ctx.state, ctx.cfg
    accept = st.decision["accept"]
    # With paired measurement the round's baseline is the fresh result, when there is a valid one.
    before = after = _evolve_now(ctx).score() if cfg.paired else st.scores[-1]
    chosen, pid, opened = None, None, False
    metrics_before = _evolve_now(ctx).metric_means()
    if accept:
        chosen = next(c for c in st.candidates if c["sample"] == accept)
        after = chosen["score"]
        workspace.fast_forward(ctx.main_wt, chosen["sha"])
        memory = _load_sim(ctx)
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
                **_measured_fields(chosen, metrics_before),
            })
            sim.save(ctx.sim_path, memory)
        problem = sim.problem(memory, pid)
        opened = (problem["opened_run"], problem["opened_round"]) == (st.run, st.round)
    _append_log(ctx, chosen, pid, before, after)
    if not st.external:
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
        # what a pull request for this edit needs
        "sha": chosen["sha"] if chosen else None,
        "title": chosen["manifest"].get("title") if chosen else None,
        "risk": chosen["manifest"]["risk"] if chosen else None,
        "expected": chosen["manifest"].get("expected", {}) if chosen else {},
        "observed": chosen.get("observed", {}) if chosen else {},
        "metrics_before": metrics_before if chosen else {},
        "metrics_after": chosen.get("metrics", {}) if chosen else {},
        "limits": chosen.get("limits", []) if chosen else [],
        "runs": cfg.rollouts,
    })
    if chosen:
        # Each accepted edit uses up part of every limit it moves.
        for status in chosen.get("limits") or []:
            name = status["metric"]
            if status["total_pct"] is not None:
                st.metric_ratio[name] = (st.metric_ratio.get(name, 1.0)
                                         * chosen["metrics"][name] / metrics_before[name])
    for c in st.candidates:
        if c.get("worktree"):
            workspace.remove_worktree(ctx.project, Path(c["worktree"]))
            workspace.delete_branch(ctx.project, c["branch"])
    if chosen:
        st.eval_sha = chosen["sha"]
        st.eval_dir = chosen.get("eval_dir", chosen["sha"])
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
    "confirm": _confirm,
    "record": _record,
    "heldout_round": _heldout_round,
    "heldout_end": _heldout_end,
    "promote": _promote,
}
