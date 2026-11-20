"""The task file, the evolve/held-out split, and tasks waiting to join the benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .state import Layout, read_json, write_json_atomic

# Ids name directories, so they start with a letter or digit: "." and ".." would alias other paths.
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TaskError(Exception):
    pass


def load_tasks(path: Path) -> list[dict]:
    if not path.is_file():
        raise TaskError(f"task file not found: {path}")
    loaded: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as error:
            raise TaskError(f"{path}:{number}: not JSON ({error.msg})") from None
        if not isinstance(obj, dict):
            raise TaskError(f"{path}:{number}: must be a JSON object")
        task_id = obj.get("id")
        if not isinstance(task_id, str) or not ID_RE.match(task_id):
            raise TaskError(f"{path}:{number}: id must match [A-Za-z0-9][A-Za-z0-9._-]*")
        if task_id in seen:
            raise TaskError(f"{path}:{number}: duplicate id {task_id}")
        seen.add(task_id)
        loaded.append(obj)
    return loaded


def assign_split(task_ids: list[str], existing: dict[str, str], heldout_fraction: float) -> dict[str, str]:
    """Give every unassigned task a side. Assigned tasks keep theirs.

    New tasks are taken in order of the SHA-256 of their id, and each goes to
    the side furthest below its target share; a tie goes to evolve.
    """
    split = dict(existing)
    present = set(task_ids)
    evolve = sum(1 for t, side in split.items() if side == "evolve" and t in present)
    heldout = sum(1 for t, side in split.items() if side == "heldout" and t in present)
    new = sorted((t for t in task_ids if t not in split), key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    for task_id in new:
        total = evolve + heldout + 1
        heldout_short = heldout_fraction * total - heldout
        evolve_short = (1 - heldout_fraction) * total - evolve
        if heldout_short > evolve_short:
            split[task_id] = "heldout"
            heldout += 1
        else:
            split[task_id] = "evolve"
            evolve += 1
    return split


def load_split(layout: Layout, task_ids: list[str], heldout_fraction: float) -> dict[str, str]:
    existing = read_json(layout.split_file, {})
    split = assign_split(task_ids, existing, heldout_fraction)
    if split != existing:
        write_json_atomic(layout.split_file, split)
    return split


def benchmark_version(layout: Layout) -> int:
    return read_json(layout.benchmark_file, {"version": 1})["version"]


def benchmark_label(name: str, layout: Layout) -> str:
    return f"{name}-v{benchmark_version(layout)}"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_pending(layout: Layout, new_tasks: list[dict], known_ids: set[str]) -> int:
    """Queue tasks from live outcomes. Returns how many were new."""
    pending = _read_jsonl(layout.pending_tasks)
    seen = known_ids | {t["id"] for t in pending}
    added = 0
    for task in new_tasks:
        if task["id"] in seen:
            continue
        seen.add(task["id"])
        pending.append(task)
        added += 1
    if added:
        layout.pending_tasks.parent.mkdir(parents=True, exist_ok=True)
        layout.pending_tasks.write_text("".join(json.dumps(t) + "\n" for t in pending))
    return added


def pending_count(layout: Layout) -> int:
    return len(_read_jsonl(layout.pending_tasks))


def fold_pending(layout: Layout, tasks_path: Path) -> int:
    """Move pending tasks into the task file and raise the benchmark version if any were added."""
    pending = _read_jsonl(layout.pending_tasks)
    if not pending:
        return 0
    current = load_tasks(tasks_path)
    have = {t["id"] for t in current}
    fresh = [t for t in pending if t["id"] not in have]
    if fresh:
        text = tasks_path.read_text()
        with tasks_path.open("a") as handle:
            if text and not text.endswith("\n"):
                handle.write("\n")  # or the first new task would join the last line
            for task in fresh:
                handle.write(json.dumps(task) + "\n")
        write_json_atomic(layout.benchmark_file, {"version": benchmark_version(layout) + 1})
    layout.pending_tasks.unlink()
    return len(fresh)
