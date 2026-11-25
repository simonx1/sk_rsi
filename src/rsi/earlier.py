"""Earlier edits that were not kept, from every run of a project, for the agents' prompts.

The memory keeps only edits that were accepted. A new run would otherwise know nothing about what
earlier runs tried and why it failed. This reads the history folders the loop already writes for every
round: each candidate's manifest and result, and the round's decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from .state import Layout

OUTCOME_CHARS = 300
TITLE_CHARS = 72
NOTES_PER_CANDIDATE = 3
UNDECIDED = "the run stopped before this round was decided"
# Decisions that compared nothing: an edit from such a round never got a fair comparison.
NOT_COMPARED = ("the round's baseline had failed runs", "selector failed:")


def _read(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):
        return None


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _round_number(folder: Path) -> int | None:
    name = folder.name
    if not folder.is_dir() or not name.startswith("round-") or not name[6:].isdigit():
        return None
    return int(name[6:])


def _title(manifest: dict) -> str:
    title = manifest.get("title") or str(manifest.get("what") or "").split("\n", 1)[0]
    return _cut(str(title), TITLE_CHARS) or "untitled"


def _guard(decision: dict | None, sample: str) -> str | None:
    """The guard that stopped this candidate after the selector chose it, if one did."""
    guards = (decision or {}).get("guards")
    if isinstance(guards, dict) and isinstance(guards.get(sample), str):
        return guards[sample]   # one per candidate: a round can stop several of the selector's choices
    reason = str((decision or {}).get("reason") or "")
    if not reason.startswith(f"guard: {sample} "):
        return None
    return reason.split("; the selector's reason was", 1)[0]


def _task_notes(table: object) -> list[str]:
    if not isinstance(table, list):
        return []
    rows = [row for row in table if isinstance(row, dict) and str(row.get("summary") or "").strip()]
    rows.sort(key=lambda row: bool(row.get("passes")))   # failing tasks first; the sort keeps the order otherwise
    return [f"{row.get('task')}: {row['summary']}" for row in rows[:NOTES_PER_CANDIDATE]]


def _outcome(result: dict, decision: dict | None, before: float | None) -> str:
    status, sample = result.get("status"), str(result.get("sample"))
    if status == "dropped":
        return f"dropped: {result.get('reason') or 'no reason given'}"
    if status != "evaluated" or not isinstance(result.get("score"), (int, float)):
        return UNDECIDED if decision is None else f"not kept ({status})"
    parts = [f"scored {result['score']:.2f}" + (f" against {before:.2f} for the version it changed"
                                                if before is not None else "")]
    observed = result.get("observed")
    changes = [f"{name} {change:+.2f}%" for name, change in (observed.items() if isinstance(observed, dict) else ())
               if isinstance(change, (int, float))]
    if changes:
        parts.append("measured " + ", ".join(changes))
    if decision is None:
        parts.append(UNDECIDED)
    else:
        reason = str(decision.get("reason") or "")
        guard = _guard(decision, sample)
        if guard:
            parts.append(guard)
        elif reason.startswith(NOT_COMPARED):
            parts.append(f"not compared: {reason}")
    parts += _task_notes(result.get("table"))
    return "; ".join(parts)


def not_kept(layout: Layout, run: str, round_no: int, limit: int = 30) -> tuple[list[dict], int]:
    """Candidates with a manifest that were not accepted, newest first, without `run`'s rounds from
    `round_no` on. Returns at most `limit` of them and how many more there are.

    Each entry: {"run", "folder" (relative to the runs folder), "sample", "title", "outcome"}.
    Unreadable files are skipped: an old or half-written folder must never stop a round.
    """
    found: list[dict] = []
    try:
        run_folders = sorted((p for p in layout.runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    except OSError:
        return [], 0
    for run_folder in run_folders:
        try:
            found += _run_entries(layout, run_folder, run, round_no)
        except Exception:  # a damaged folder is skipped, never a reason to stop a round
            continue
    return found[:limit], max(len(found) - limit, 0)


def _run_entries(layout: Layout, run_folder: Path, run: str, round_no: int) -> list[dict]:
    history = run_folder / "history"
    if not history.is_dir():
        return []
    state = _read(layout.run(run_folder.name).state_file)
    log = state.get("rounds_log") if isinstance(state, dict) else None
    before = {entry.get("round"): entry.get("score_before") for entry in (log if isinstance(log, list) else ())
              if isinstance(entry, dict)}
    rounds = sorted(((n, p) for p in history.iterdir() if (n := _round_number(p)) is not None), reverse=True)
    entries = []
    for number, round_folder in rounds:
        if run_folder.name == run and number >= round_no:
            continue
        try:
            folders = sorted(p for p in round_folder.iterdir() if p.is_dir())
        except OSError:
            continue
        decision = _read(round_folder / "decision.json")
        decision = decision if isinstance(decision, dict) else None
        score_before = before.get(number)
        score_before = score_before if isinstance(score_before, (int, float)) else None
        for folder in folders:
            try:
                entry = _entry(layout.runs_dir, folder, decision, score_before)
            except Exception:
                continue
            if entry is not None:
                entries.append(entry)
    return entries


def _entry(runs_dir: Path, folder: Path, decision: dict | None, score_before: float | None) -> dict | None:
    manifest, result = _read(folder / "manifest.json"), _read(folder / "result.json")
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        return None
    if decision is not None and decision.get("accept") == folder.name:
        return None   # kept: it is in the memory
    return {
        "run": folder.parent.parent.parent.name,
        "folder": folder.relative_to(runs_dir).as_posix(),
        "sample": folder.name,
        "title": _title(manifest),
        "outcome": _cut(_outcome(result, decision, score_before), OUTCOME_CHARS),
    }
