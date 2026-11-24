"""The live progress line: what a run is doing, how far it is, and whether anything failed.

Its content is a template with named fields, set under [progress] in a project's config.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

DEFAULT_LINE = ("round {round}/{rounds} · {step} · runs {done}/{total}, {failed} failed · agents {agents} · "
                "score {score} · {step_elapsed} in step, {elapsed} total, ~{eta} left")

FIELDS = frozenset({"run", "status", "round", "rounds", "step", "done", "total", "failed", "agents", "errors",
                    "accepted", "score", "start_score", "elapsed", "step_elapsed", "eta"})
GROUPS = ("metric", "limit")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def check_template(line: str) -> None:
    """Refuse a template that names a field the progress line does not have."""
    try:
        parts = list(string.Formatter().parse(line))
    except ValueError as error:
        raise ValueError(f"progress.line is not a valid template: {error}") from None
    for _, name, _, _ in parts:
        if name is None:
            continue
        group, dot, rest = name.partition(".")
        ok = (not dot and name in FIELDS) or (dot and group in GROUPS and NAME.match(rest))
        if not ok:
            raise ValueError(f"unknown field {name or '{}'} in progress.line")


def duration(seconds: float) -> str:
    """45s, 38m, 4h12m."""
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


@dataclass
class Snapshot:
    """Everything the line can show, at one moment."""
    run: str
    status: str = "running"
    round: int = 0
    rounds: int = 0
    step: str = "starting"
    done: int = 0
    total: int = 0
    failed: int = 0
    agents_running: int = 0
    agents_total: int = 0
    errors: int = 0
    accepted: int = 0
    score: float | None = None
    start_score: float | None = None
    metrics: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    elapsed: float = 0.0
    step_elapsed: float = 0.0
    eta: float | None = None


class _Group:
    """`metric.<name>` and `limit.<name>`: any name, `-` when there is no value."""

    def __init__(self, values: dict, show):
        self._values, self._show = values, show

    def __getattr__(self, name: str) -> str:
        value = self._values.get(name)
        return "-" if value is None else self._show(value)


def render(line: str, snap: Snapshot) -> str:
    number = lambda value: "-" if value is None else f"{value:.2f}"
    values = {
        "run": snap.run, "status": snap.status, "step": snap.step,
        "round": str(snap.round), "rounds": str(snap.rounds), "done": str(snap.done), "total": str(snap.total),
        "failed": str(snap.failed), "errors": str(snap.errors), "accepted": str(snap.accepted),
        "agents": f"{snap.agents_running}/{snap.agents_total}" if snap.agents_total else "-",
        "score": number(snap.score), "start_score": number(snap.start_score),
        "elapsed": duration(snap.elapsed), "step_elapsed": duration(snap.step_elapsed),
        "eta": "?" if snap.eta is None else duration(snap.eta),
        "metric": _Group(snap.metrics, lambda v: f"{v:.2f}"),
        "limit": _Group(snap.limits, lambda v: f"{v:+.2f}%"),
    }
    return line.format_map(values)


def eta_seconds(runs_left: int, run_seconds: list[float], parallel: int,
                agent_rounds_left: int, agent_seconds: list[float]) -> float | None:
    """Time left: runs still to make at the average run time so far, plus agent time per round still to come."""
    if not run_seconds:
        return None
    runs = runs_left * (sum(run_seconds) / len(run_seconds)) / max(parallel, 1)
    agents = agent_rounds_left * (sum(agent_seconds) / len(agent_seconds)) if agent_seconds else 0.0
    return runs + agents
