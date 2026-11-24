"""Push the run branch and open a pull request that can be judged from its body alone."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import measure, sim, workspace
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


def promote(project: Path, st: RunState, cfg: Config, memory: dict, scratch: Path | None = None) -> dict:
    """Open the pull request for a finished run, or say why not and how to do it by hand.

    With promotion.mode = "per_edit" it opens one pull request per accepted edit instead; `scratch`
    is where the branches for them are built.
    """
    if not _accepted(st):
        return {"skipped": "no edit was accepted in this run"}
    if cfg.promotion_mode == "per_edit":
        return _promote_per_edit(project, st, cfg, memory, scratch or Path(tempfile.mkdtemp(prefix="rsi-promote-")) / "wt")
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


# --- one pull request per accepted edit ---------------------------------------

def edit_branch(cfg: Config, st: RunState, entry: dict) -> str:
    return f"{cfg.branch_prefix}{st.run}-r{entry['round']}"


def build_edit_branches(project: Path, st: RunState, cfg: Config, scratch: Path) -> list[dict]:
    """A branch per accepted edit, in round order.

    The edit is cherry-picked onto the run's starting commit, so its pull request shows that edit
    alone. An edit that does not apply there depends on earlier ones: its branch is then the edit's
    own commit on the run branch, which contains them ("stacked"). A branch that exists is kept.
    """
    built = []
    for entry in _accepted(st):
        branch = edit_branch(cfg, st, entry)
        if not workspace.branch_exists(project, branch):
            workspace.add_detached_worktree(project, scratch, st.start_sha)
            clean = workspace.cherry_pick(scratch, entry["sha"])
            tip = workspace.rev_parse(scratch) if clean else entry["sha"]
            workspace.remove_worktree(project, scratch)
            workspace.set_branch(project, branch, tip)
        stacked = workspace.rev_parse(project, branch) == entry["sha"] and entry is not _accepted(st)[0]
        built.append({"round": entry["round"], "branch": branch, "stacked": stacked})
    return built


def run_warning(st: RunState) -> str | None:
    """Why this run's pull requests should not be merged without a closer look, if there is a reason."""
    if _heldout_dropped(st):
        return f"the held-out score went down ({st.heldout['start']:.2f} → {st.heldout['end']:.2f})"
    blocked = measure.blocking(st.heldout.get("limits") or [])
    if blocked:
        return f"the whole run is past a limit: {measure.describe(blocked[0])}"
    return None


def edit_title(entry: dict) -> str:
    if entry.get("title"):
        return entry["title"]
    stripped = entry["what"].strip()
    line = stripped.splitlines()[0] if stripped else f"RSI edit of round {entry['round']}"
    return line if len(line) <= 72 else line[:69] + "..."


def _signed(changes: dict) -> str:
    return ", ".join(f"{name} {change:+.2f}%" for name, change in changes.items())


def edit_body(st: RunState, cfg: Config, memory: dict, entry: dict, earlier: list[dict], stacked: bool) -> str:
    """The pull request text for one accepted edit: what, the measurements, and what they were taken on."""
    lines: list[str] = []
    warning = run_warning(st)
    if warning:
        lines += [f"> **Warning:** {warning}. Review this run before merging any of its pull requests.", ""]
    lines += ["## What", "", entry["what"].strip(), "", "## Measured", "",
              "| | Before | After | Change |", "|---|---|---|---|",
              f"| Score | {entry['score_before']:.2f} | {entry['score_after']:.2f} | "
              f"{entry['score_after'] - entry['score_before']:+.2f} |"]
    before, after = entry.get("metrics_before") or {}, entry.get("metrics_after") or {}
    observed = entry.get("observed") or {}
    for name in after:
        if name in before:
            change = f"{observed[name]:+.2f}%" if name in observed else ""
            lines.append(f"| {name} | {before[name]:.2f} | {after[name]:.2f} | {change} |")
    lines.append("")
    runs = f"{entry['runs']} run{'' if entry['runs'] == 1 else 's'} per task"
    if cfg.paired:
        lines.append(f"Each side: {runs}, measured in the same batch in round {entry['round']}.")
    else:
        lines.append(f"Each side: {runs}. The baseline was measured before round {entry['round']}.")
    lines += ["", "## Predicted and measured", ""]
    predicted = [f"- {name}: predicted {change:+.2f}%, measured "
                 + (f"{observed[name]:+.2f}%" if name in observed else "not reported")
                 for name, change in (entry.get("expected") or {}).items()]
    if entry["fixed"] or entry["missed"]:
        predicted.append(f"- Tasks predicted to pass: fixed {_names(entry['fixed'])}; missed {_names(entry['missed'])}")
    lines += predicted or ["The proposer made no prediction."]
    lines += ["", "## Risk", "", (entry.get("risk") or "").strip() or "None named.", "", "## Measured on top of", ""]
    if earlier:
        lines += [f"- {edit_title(e)}" for e in earlier]
        lines += ["", "The numbers above compare this edit with a version that already had those edits."]
        if stacked:
            lines.append("This branch contains those edits. Merge them first.")
    else:
        lines.append("Nothing: this edit was measured on the starting version.")
    lines += ["", "## Whole run", "",
              f"Run {st.run} on {st.benchmark}. Held-out: {st.heldout['start']:.2f} → {st.heldout['end']:.2f}"]
    lines += [f"- {measure.describe(status)}" for status in st.heldout.get("limits") or []]
    lines += ["", "## Memory", ""]
    problem = sim.problem(memory, entry["problem"]) if entry.get("problem") else None
    name = problem["name"] if problem else "(not found in the memory file)"
    lines.append(f"**{entry.get('problem')}**: {name} (run {st.run}, round {entry['round']}).")
    return "\n".join(lines) + "\n"


def _existing_pr(project: Path, branch: str) -> dict | None:
    """Our own pull request for this branch, if there is one, so that promoting twice makes no duplicate.

    A pull request with the same branch name is ours only when it comes from this repository and its
    head is the commit we pushed. One from a fork, or for another commit, is somebody else's.
    """
    done = subprocess.run(["gh", "pr", "list", "--head", branch, "--state", "all",
                           "--json", "url,isDraft,headRefOid,isCrossRepository", "--limit", "20"],
                          cwd=str(project), capture_output=True, text=True)
    if done.returncode != 0:
        return None
    try:
        found = json.loads(done.stdout or "[]")
    except ValueError:
        return None
    tip = workspace.rev_parse(project, branch)
    for pr in found if isinstance(found, list) else []:
        if isinstance(pr, dict) and not pr.get("isCrossRepository") and pr.get("headRefOid") == tip:
            return pr
    return None


def _create_pr(project: Path, base: str, branch: str, title: str, body: str, draft: bool) -> dict:
    handle, body_path = tempfile.mkstemp(prefix="rsi-pr-", suffix=".md")
    try:
        with os.fdopen(handle, "w") as fh:
            fh.write(body)
        args = ["gh", "pr", "create", "--base", base, "--head", branch, "--title", title, "--body-file", body_path]
        if draft:
            args.append("--draft")
        done = subprocess.run(args, cwd=str(project), capture_output=True, text=True)
    finally:
        os.unlink(body_path)
    if done.returncode != 0:
        return {"error": f"gh pr create failed: {done.stderr.strip()}"}
    printed = done.stdout.strip().splitlines()
    return {"url": printed[-1] if printed else "", "draft": draft}


def _promote_per_edit(project: Path, st: RunState, cfg: Config, memory: dict, scratch: Path) -> dict:
    accepted = {entry["round"]: entry for entry in _accepted(st)}
    branches = build_edit_branches(project, st, cfg, scratch)
    kept = {r["round"]: r for r in (st.promoted or {}).get("prs", []) if r.get("url")}
    remotes = workspace.remotes(project)
    blocked = None
    if not cfg.pull_request:
        blocked = "promotion.pull_request is false"
    elif not remotes:
        blocked = "the project has no git remote"
    elif shutil.which("gh") is None:
        blocked = "gh is not installed"
    remote = "origin" if "origin" in remotes else (remotes[0] if remotes else "origin")
    draft = cfg.draft or run_warning(st) is not None
    records, manual = [], []
    for built in branches:
        if built["round"] in kept:  # opened by an earlier call
            records.append(kept[built["round"]])
            continue
        entry, branch = accepted[built["round"]], built["branch"]
        title = edit_title(entry)
        record = dict(built)
        if blocked:
            record["skipped"] = blocked
            manual += [f"git push -u {remote} {branch}",
                       f"gh pr create --base {cfg.base_branch} --head {branch} --title {shlex.quote(title)}"
                       + (" --draft" if draft else "")]
        else:
            try:
                workspace.git(project, "push", "-u", remote, branch)
            except workspace.GitError as error:
                record["error"] = f"push failed: {error}"
            else:
                existing = _existing_pr(project, branch)
                if existing:
                    record.update(url=existing.get("url", ""), draft=bool(existing.get("isDraft")))
                else:
                    earlier = [e for r, e in accepted.items() if r < entry["round"]]
                    body = edit_body(st, cfg, memory, entry, earlier, built["stacked"])
                    record.update(_create_pr(project, cfg.base_branch, branch, title, body, draft))
        records.append(record)
    result: dict = {"prs": records}
    if manual:
        result["manual"] = manual
    return result
