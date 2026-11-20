"""Push the run branch and open a pull request that can be judged from its body alone."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import sim, workspace
from .config import Config
from .state import RunState


def _accepted(st: RunState) -> list[dict]:
    return [entry for entry in st.rounds_log if entry["accepted"]]


def _heldout_dropped(st: RunState) -> bool:
    return st.heldout["end"] < st.heldout["start"]


def _names(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def pr_title(st: RunState) -> str:
    return (
        f"RSI {st.run}: evolve {st.scores[0]:.2f} → {st.scores[-1]:.2f}, "
        f"held-out {st.heldout['start']:.2f} → {st.heldout['end']:.2f}"
    )


def pr_body(st: RunState, cfg: Config, memory: dict) -> str:
    start, end = st.heldout["start"], st.heldout["end"]
    lines: list[str] = []
    if _heldout_dropped(st):
        lines += [
            f"> **Warning:** the held-out score went down ({start:.2f} → {end:.2f}). "
            "The gain on the evolve set may not generalise. Review before merging.",
            "",
        ]
    lines += [
        "## Setting",
        "",
        f"- Benchmark: {st.benchmark}",
        f"- Model: {cfg.model}",
        f"- Run: {st.run}, {len(st.rounds_log)} rounds",
        "",
        "## Scores",
        "",
        "| Set | Start | End |",
        "|---|---|---|",
        f"| Evolve | {st.scores[0]:.2f} | {st.scores[-1]:.2f} |",
        f"| Held-out | {start:.2f} | {end:.2f} |",
        "",
        "## Rounds",
        "",
        "| Round | Accepted edit | Evolve | Fixed | Missed | Regressed |",
        "|---|---|---|---|---|---|",
    ]
    for entry in st.rounds_log:
        if entry["accepted"]:
            edit = f"{entry['accepted']} ({entry['problem']}): {_cell(entry['what'])}"
        else:
            edit = f"none: {_cell(entry['reason'])}"
        lines.append(
            f"| {entry['round']} | {edit} | {entry['score_before']:.2f} → {entry['score_after']:.2f} | "
            f"{_names(entry['fixed'])} | {_names(entry['missed'])} | {_names(entry['regressed'])} |"
        )
    lines += ["", "## Memory", ""]
    accepted = _accepted(st)
    for pid in dict.fromkeys(entry["problem"] for entry in accepted):
        mine = [entry for entry in accepted if entry["problem"] == pid]
        entry_in_memory = sim.problem(memory, pid)
        name = entry_in_memory["name"] if entry_in_memory else "(not found in the memory file)"
        total = len(entry_in_memory["attempts"]) if entry_in_memory else 0
        verb = "opened" if any(entry["opened"] for entry in mine) else "extended"
        noun = "attempt" if len(mine) == 1 else "attempts"
        lines.append(f"- **{pid}** ({verb}): {name}. {len(mine)} {noun} in this run, {total} in total.")
    lines += [
        "",
        "Held-out tasks, their answers and their results were never shown to the optimizer.",
        "The full record is in `.rsi/sim.json` and `.rsi/log.md` on this branch.",
    ]
    return "\n".join(lines) + "\n"


def _manual(st: RunState, cfg: Config) -> list[str]:
    return [
        f"git push -u origin {st.branch}",
        f"gh pr create --base {cfg.base_branch} --head {st.branch} --title {shlex.quote(pr_title(st))}",
    ]


def promote(project: Path, st: RunState, cfg: Config, memory: dict) -> dict:
    """Open the pull request for a finished run, or say why not and how to do it by hand."""
    if not _accepted(st):
        return {"skipped": "no edit was accepted in this run"}
    manual = _manual(st, cfg)
    if not cfg.pull_request:
        return {"skipped": "promotion.pull_request is false", "manual": manual}
    names = workspace.remotes(project)
    if not names:
        return {"skipped": "the project has no git remote", "manual": manual}
    if shutil.which("gh") is None:
        return {"skipped": "gh is not installed", "manual": manual}
    remote = "origin" if "origin" in names else names[0]
    try:
        workspace.git(project, "push", "-u", remote, st.branch)
    except workspace.GitError as error:
        return {"error": f"push failed: {error}", "manual": manual}
    draft = _heldout_dropped(st)
    handle, body_path = tempfile.mkstemp(prefix="rsi-pr-", suffix=".md")
    try:
        with os.fdopen(handle, "w") as fh:
            fh.write(pr_body(st, cfg, memory))
        args = ["gh", "pr", "create", "--base", cfg.base_branch, "--head", st.branch,
                "--title", pr_title(st), "--body-file", body_path]
        if draft:
            args.append("--draft")
        done = subprocess.run(args, cwd=str(project), capture_output=True, text=True)
    finally:
        os.unlink(body_path)
    if done.returncode != 0:
        return {"error": f"gh pr create failed: {done.stderr.strip()}", "manual": manual}
    printed = done.stdout.strip().splitlines()
    return {"url": printed[-1] if printed else "", "draft": draft}
