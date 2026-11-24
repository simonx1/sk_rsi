"""Percent changes of metrics, and limits on them.

A limit is relative to the harness a run started from and it is cumulative: `used` holds, per metric,
the product of candidate/baseline over the rounds whose edit was accepted. Everything is compared and
shown at two decimals.
"""

from __future__ import annotations

from typing import Iterable

from .config import Constraint


def pct_change(before: float, after: float) -> float | None:
    """The change from `before` to `after` in percent, or None when there is nothing to divide by."""
    if not before:
        return None
    return round((after / before - 1) * 100, 2)


def changes(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Percent change of every metric both sides report."""
    found = {}
    for name, value in before.items():
        if name in after:
            change = pct_change(value, after[name])
            if change is not None:
                found[name] = change
    return found


def limit_status(constraints: Iterable[Constraint], used: dict[str, float],
                 base: dict[str, float], cand: dict[str, float]) -> list[dict]:
    """Where a candidate stands against each limit, counting the rounds accepted before it.

    States: "ok"; "over" (past the limit); "missing" (the baseline reports the metric and the
    candidate does not: a guard that was not measured does not count as passed); "not measured"
    (the baseline does not report it, or reports zero).
    """
    statuses = []
    for c in constraints:
        status = {"metric": c.metric, "direction": c.direction, "allowed_pct": c.pct, "total_pct": None}
        if not base.get(c.metric):
            status["state"] = "not measured"
        elif c.metric not in cand:
            status["state"] = "missing"
        else:
            total = used.get(c.metric, 1.0) * cand[c.metric] / base[c.metric]
            status["total_pct"] = round((total - 1) * 100, 2)
            moved = status["total_pct"] if c.direction == "increase" else -status["total_pct"]
            status["state"] = "over" if moved > c.pct else "ok"
        statuses.append(status)
    return statuses


def used_status(constraints: Iterable[Constraint], used: dict[str, float]) -> list[dict]:
    """How much of each limit the accepted rounds have used so far."""
    return [{"metric": c.metric, "direction": c.direction, "allowed_pct": c.pct,
             "total_pct": round((used.get(c.metric, 1.0) - 1) * 100, 2), "state": "ok"} for c in constraints]


def blocking(statuses: list[dict]) -> list[dict]:
    """The statuses that stop a candidate from being accepted."""
    return [s for s in statuses if s["state"] in ("over", "missing")]


def describe(status: dict) -> str:
    if status["state"] == "missing":
        return f"{status['metric']} was not reported by the candidate"
    if status["state"] == "not measured":
        return f"{status['metric']} not measured"
    sign = "+" if status["direction"] == "increase" else "-"
    return f"{status['metric']} {status['total_pct']:+.2f}% of {sign}{status['allowed_pct']:.2f}% allowed"
