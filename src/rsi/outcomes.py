"""Live outcomes: evidence for the prompts, and new tasks waiting to join the benchmark.

Live outcomes never decide which edit is accepted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import shell, tasks
from .config import Config
from .state import Layout, read_json, write_json_atomic


class OutcomeError(Exception):
    pass


def _problem(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return "must be a JSON object"
    if obj.get("kind") not in ("evidence", "task"):
        return 'kind must be "evidence" or "task"'
    if not isinstance(obj.get("id"), str) or not obj["id"]:
        return "id must be non-empty text"
    if not isinstance(obj.get("summary"), str) or not obj["summary"]:
        return "summary must be non-empty text"
    if obj["kind"] == "task":
        task = obj.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not tasks.ID_RE.match(task["id"]):
            return "a task record needs a task object whose id matches [A-Za-z0-9][A-Za-z0-9._-]*"
    elif "task_id" in obj and not isinstance(obj["task_id"], str):
        return "task_id must be text"
    return None


def parse_records(text: str) -> tuple[list[dict], list[str]]:
    """Split ingest output into valid records and a note for each line that was skipped."""
    records, problems = [], []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not JSON")
            continue
        problem = _problem(obj)
        if problem:
            problems.append(f"line {number}: {problem}")
        else:
            records.append(obj)
    return records, problems


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def ingest(cfg: Config, layout: Layout, project: Path, now: datetime | None = None) -> dict:
    """Run the project's ingest command and store whatever is new."""
    if not cfg.ingest:
        return {"evidence": 0, "tasks": 0, "skipped": []}
    since = layout.ingest_stamp.read_text().strip() if layout.ingest_stamp.is_file() else ""
    started = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = shell.run(shell.render(cfg.ingest, {"since": since}), cwd=project, timeout=cfg.task_timeout_s)
    if done.timed_out:
        raise OutcomeError(f"ingest timed out after {cfg.task_timeout_s}s")
    if done.returncode != 0:
        raise OutcomeError(f"ingest failed with exit code {done.returncode}: {done.stderr.strip()[-500:]}")
    records, problems = parse_records(done.stdout)
    tasks_path = layout.tasks_path(cfg.tasks_file)
    try:
        known = {t["id"] for t in tasks.load_tasks(tasks_path)} if tasks_path.is_file() else set()
    except tasks.TaskError as error:
        raise OutcomeError(f"ingest could not read the task file: {error}") from None
    seen = set(_lines(layout.seen_ids))
    fresh = []
    for record in records:
        if record["id"] not in seen:
            seen.add(record["id"])
            fresh.append(record)
    evidence = [r for r in fresh if r["kind"] == "evidence"]
    new_tasks = [r["task"] for r in fresh if r["kind"] == "task"]
    layout.outcomes.mkdir(parents=True, exist_ok=True)
    if evidence:
        with layout.evidence_file.open("a") as fh:
            for record in evidence:
                fh.write(json.dumps(record) + "\n")
    added = tasks.add_pending(layout, new_tasks, known)
    if fresh:
        with layout.seen_ids.open("a") as fh:
            for record in fresh:
                fh.write(record["id"] + "\n")
    layout.ingest_stamp.write_text(started + "\n")
    return {"evidence": len(evidence), "tasks": added, "skipped": problems}


def new_evidence(layout: Layout, evolve_ids: set[str]) -> tuple[list[dict], int]:
    """Evidence not yet shown to a round that an agent may see.

    A record is shown when it names no task or names an evolve task. One that names any other
    task is withheld: that task is held-out, or queued, or gone from the benchmark, and may be
    or become held-out. Returns the records and the cursor to hand to mark_shown once saved.
    """
    lines = _lines(layout.evidence_file)
    shown = read_json(layout.evidence_cursor, {"shown": 0})["shown"]
    records = [json.loads(line) for line in lines[shown:]]
    return [r for r in records if "task_id" not in r or r["task_id"] in evolve_ids], len(lines)


def mark_shown(layout: Layout, cursor: int) -> None:
    write_json_atomic(layout.evidence_cursor, {"shown": cursor})
