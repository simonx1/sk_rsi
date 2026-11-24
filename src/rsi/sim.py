"""The Self-Improvement Memory: root causes, and every accepted attempt at each.

Shape (Yu et al., arXiv:2611.18856, Appendix D.3, with additions):

    {"next_id": int, "carried_from": [hop], "aliases": {old_id: id},
     "problems": [{"id", "name", "opened_round", "opened_run", "attempts": [attempt]}]}

Only the framework writes this file. Attempts are appended, never deleted.
"""

from __future__ import annotations

import json
from pathlib import Path

from .state import write_json_atomic


class SimError(Exception):
    pass


def empty() -> dict:
    return {"next_id": 1, "carried_from": [], "aliases": {}, "problems": []}


def load(path: Path) -> dict:
    if not path.is_file():
        return empty()
    try:
        memory = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SimError(f"{path} is not valid JSON: {error.msg}") from None
    shape = {"next_id": int, "carried_from": list, "aliases": dict, "problems": list}
    for key, kind in shape.items():
        if not isinstance(memory, dict) or not isinstance(memory.get(key), kind):
            raise SimError(f"{path}: '{key}' is missing or has the wrong type")
    return memory


def save(path: Path, memory: dict) -> None:
    write_json_atomic(path, memory)


def _norm(text: str) -> str:
    return " ".join(text.split())


def problem(memory: dict, pid: str) -> dict | None:
    return next((p for p in memory["problems"] if p["id"] == pid), None)


def resolve(memory: dict, pid: str) -> str | None:
    """Follow aliases to the surviving problem id, or None when there is none."""
    seen: set[str] = set()
    while pid in memory["aliases"] and pid not in seen:
        seen.add(pid)
        pid = memory["aliases"][pid]
    return pid if problem(memory, pid) else None


def find_attempt(memory: dict, run: str, round_no: int) -> str | None:
    for p in memory["problems"]:
        if any(a["run"] == run and a["round"] == round_no for a in p["attempts"]):
            return p["id"]
    return None


def record_attempt(memory: dict, root_cause: dict, attempt: dict) -> str:
    """Append an attempt under its root cause and return the problem id.

    {"id": ...} must name an existing problem (aliases are followed).
    {"new": text} appends to a problem with the same name, or opens one.
    """
    if "id" in root_cause:
        pid = resolve(memory, root_cause["id"])
        if pid is None:
            raise SimError(f"root cause {root_cause['id']} is not in the memory")
        target = problem(memory, pid)
    else:
        name = _norm(root_cause["new"])
        target = next((p for p in memory["problems"] if _norm(p["name"]) == name), None)
        if target is None:
            target = {
                "id": f"P{memory['next_id']}",
                "name": name,
                "opened_round": attempt["round"],
                "opened_run": attempt["run"],
                "attempts": [],
            }
            memory["next_id"] += 1
            memory["problems"].append(target)
    target["attempts"].append(attempt)
    return target["id"]


def merge(memory: dict, src: str, dst: str) -> None:
    """Move every attempt of `src` into `dst`, in time order, and keep an alias."""
    src_id, dst_id = resolve(memory, src), resolve(memory, dst)
    if src_id is None or dst_id is None:
        raise SimError(f"unknown problem: {src if src_id is None else dst}")
    if src_id == dst_id:
        raise SimError(f"{src} and {dst} are the same problem")
    source, target = problem(memory, src_id), problem(memory, dst_id)
    target["attempts"] = sorted(target["attempts"] + source["attempts"], key=lambda a: (a["run"], a["round"]))
    memory["problems"].remove(source)
    for old, new in list(memory["aliases"].items()):
        if new == src_id:
            memory["aliases"][old] = dst_id
    memory["aliases"][src_id] = dst_id


def add_hop(memory: dict, hop: dict) -> None:
    memory["carried_from"].append(hop)


def latest_attempt(memory: dict) -> dict | None:
    attempts = [a for p in memory["problems"] for a in p["attempts"]]
    return max(attempts, key=lambda a: (a["run"], a["round"]), default=None)


def outcome(predicted: list[str], before: set[str], after: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Compare predictions with what passed before and after an edit."""
    fixed = [t for t in predicted if t not in before and t in after]
    missed = [t for t in predicted if t not in after]
    regressed = sorted(before - after)
    return fixed, missed, regressed


def _names(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def render_text(memory: dict) -> str:
    if not memory["problems"]:
        return "The memory is empty."
    lines: list[str] = []
    for hop in memory["carried_from"]:
        lines.append(f"Carried from run {hop['run']} ({hop['benchmark']}, {hop['model']}) into {hop['into']}")
    for p in memory["problems"]:
        lines.append("")
        lines.append(f"{p['id']}  {p['name']}")
        lines.append(f"    opened in run {p['opened_run']}, round {p['opened_round']}")
        for a in p["attempts"]:
            domain = a["domain"]
            lines.append(f"  - {a['run']} round {a['round']} ({domain['benchmark']}, {domain['model']}): {a['what']}")
            lines.append(
                f"    predicted {_names(a['predicted'])}; fixed {_names(a['fixed'])}; "
                f"missed {_names(a['missed'])}; regressed {_names(a['regressed'])}"
            )
            lines.append(
                f"    evolve {a['score_before']:.2f} -> {a['score_after']:.2f}, "
                f"{a['passed']} of {a['total']} rollouts succeeded"
            )
            parts = [f"{label} " + ", ".join(f"{name} {change:+.2f}%" for name, change in a[key].items())
                     for key, label in (("expected", "expected"), ("observed", "measured")) if a.get(key)]
            if parts:
                lines.append("    " + "; ".join(parts))
    for old, new in memory["aliases"].items():
        lines.append(f"{old} was merged into {new}")
    return "\n".join(lines).strip()
