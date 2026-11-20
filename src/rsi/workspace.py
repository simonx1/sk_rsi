"""Git worktrees and branches in the target project, and the rules for what a candidate may change."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


class GitError(Exception):
    pass


def git(cwd: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and done.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def git_ok(cwd: Path, *args: str) -> bool:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True).returncode == 0


def repo_root(path: Path) -> Path:
    return Path(git(path, "rev-parse", "--show-toplevel"))


def rev_parse(cwd: Path, ref: str = "HEAD") -> str:
    return git(cwd, "rev-parse", ref)


def show(repo: Path, ref: str, path: str) -> str:
    """The content of `path` as committed at `ref`."""
    return git(repo, "show", f"{ref}:{path}")


# --- path rules -----------------------------------------------------------

def _glob_to_regex(pattern: str) -> re.Pattern:
    """`*` and `?` stay within one path segment; `**` crosses segments."""
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            parts.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(parts) + r"\Z")


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(_glob_to_regex(pattern).match(path) for pattern in patterns)


def path_violations(paths: list[str], editable: Iterable[str], frozen: Iterable[str]) -> list[str]:
    """One line per changed path a candidate was not allowed to touch."""
    editable, frozen = list(editable), list(frozen)
    found = []
    for path in paths:
        if path == ".rsi" or path.startswith(".rsi/"):
            found.append(f"{path}: only the framework writes .rsi/")
        elif matches_any(path, frozen):
            found.append(f"{path}: frozen")
        elif not matches_any(path, editable):
            found.append(f"{path}: not editable")
    return found


def command_in_editable(command: str, repo: Path, editable: Iterable[str], frozen: Iterable[str]) -> bool:
    """True when the command's first word is a repo file that a candidate could rewrite."""
    try:
        first = Path(shlex.split(command)[0])
    except (ValueError, IndexError):
        return False
    if first.is_absolute():
        try:
            first = first.resolve().relative_to(repo.resolve())
        except ValueError:
            return False
    if not (repo / first).is_file():
        return False
    relative = first.as_posix()
    return matches_any(relative, editable) and not matches_any(relative, frozen)


# --- branches and worktrees -------------------------------------------------

def branch_exists(repo: Path, branch: str) -> bool:
    return git_ok(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")


def delete_branch(repo: Path, branch: str) -> None:
    git(repo, "branch", "-D", branch, check=False)


def remove_worktree(repo: Path, path: Path) -> None:
    if path.exists():
        git(repo, "worktree", "remove", "--force", str(path), check=False)
        shutil.rmtree(path, ignore_errors=True)
    git(repo, "worktree", "prune", check=False)


def add_worktree(repo: Path, path: Path, branch: str, start: str) -> None:
    """Check `branch` out at `path`. The branch is created at `start` when it does not exist."""
    remove_worktree(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if branch_exists(repo, branch):
        git(repo, "worktree", "add", str(path), branch)
    else:
        git(repo, "worktree", "add", "-b", branch, str(path), start)


def reset_worktree(repo: Path, path: Path, branch: str, start: str) -> None:
    """A fresh worktree on a fresh `branch` at `start`, discarding whatever was there."""
    remove_worktree(repo, path)
    delete_branch(repo, branch)
    add_worktree(repo, path, branch, start)


def commit_all(worktree: Path, message: str, paths: list[str] | None = None) -> str | None:
    """Commit every change (or only those under `paths`). Returns the new commit, or None when there is nothing."""
    git(worktree, "add", "-A", "--", *(paths or ["."]))
    if git_ok(worktree, "diff", "--cached", "--quiet"):
        return None
    git(worktree, "commit", "--no-verify", "-m", message)
    return rev_parse(worktree)


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    """Paths that differ between two commits. A rename counts as its old and its new path."""
    out = git(repo, "diff", "--name-only", "--no-renames", base, head)
    return sorted(line for line in out.splitlines() if line)


def diff_text(repo: Path, base: str, head: str) -> str:
    return git(repo, "diff", "--no-renames", base, head)


def fast_forward(worktree: Path, sha: str) -> None:
    git(worktree, "merge", "--ff-only", sha)


def is_merged(repo: Path, branch: str, into: str) -> bool:
    return git_ok(repo, "merge-base", "--is-ancestor", branch, into)


def remotes(repo: Path) -> list[str]:
    return [line for line in git(repo, "remote").splitlines() if line]
