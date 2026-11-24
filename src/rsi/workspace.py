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


# Agents can write in the worktrees the loop runs git in. These settings keep a git config an agent
# planted from running a command (an fsmonitor) and keep repository hooks out of the loop's own git calls.
# The real guard is require_gitfile(): a worktree whose .git was replaced is refused.
SAFE = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null")


def git(cwd: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(["git", *SAFE, *args], cwd=str(cwd), capture_output=True, text=True)
    if check and done.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def git_ok(cwd: Path, *args: str) -> bool:
    return subprocess.run(["git", *SAFE, *args], cwd=str(cwd), capture_output=True).returncode == 0


def read_gitfile(worktree: Path) -> str | None:
    """The text of a linked worktree's .git file, or None when .git is not a plain file."""
    path = worktree / ".git"
    if path.is_symlink() or not path.is_file():
        return None
    return path.read_text()


def require_gitfile(worktree: Path) -> None:
    """Refuse a worktree whose .git does not point at a worktree record of a repository outside it.

    A .git that an agent replaced could point git at a configuration the agent wrote.
    """
    text = read_gitfile(worktree)
    target = Path(text.strip()[len("gitdir:"):].strip()) if text and text.startswith("gitdir:") else None
    if (target is None or not target.is_absolute() or target.parent.name != "worktrees"
            or target.resolve().is_relative_to(worktree.resolve())):
        raise GitError(f"{worktree}/.git is not the worktree's own .git file; refusing to run git there")


def restore_gitfile(worktree: Path, text: str) -> bool:
    """Put a worktree's .git file back as it was. Returns True when it had been changed."""
    if read_gitfile(worktree) == text:
        return False
    path = worktree / ".git"
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(text)
    return True


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


def _named_words(command: str) -> list[str]:
    """Every word of a command line, plus the words inside quoted arguments and after `NAME=`."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        words = command.split()
    found = list(words)
    for word in words:
        if any(ch.isspace() for ch in word):  # e.g. the argument of `bash -c '...'`
            try:
                found += shlex.split(word)
            except ValueError:
                found += word.split()
        if "=" in word:
            found.append(word.split("=", 1)[1])
    return found


def command_in_editable(command: str, repo: Path, editable: Iterable[str], frozen: Iterable[str]) -> bool:
    """True when the command names a repo file that a candidate could rewrite.

    It fails closed: the file counts wherever it appears, because working out which words a
    shell command really executes cannot be done reliably.
    """
    editable, frozen = list(editable), list(frozen)
    root = repo.resolve()
    for word in _named_words(command):
        try:
            full = (root / word).resolve()  # an absolute word replaces root; ".." is normalised
            relative = full.relative_to(root).as_posix()
        except (ValueError, OSError):
            continue
        if full.is_file() and matches_any(relative, editable) and not matches_any(relative, frozen):
            return True
    return False


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


def add_detached_worktree(repo: Path, path: Path, ref: str) -> None:
    """A worktree at `ref` that is on no branch, replacing whatever was at `path`."""
    remove_worktree(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "--detach", str(path), ref)


def reset_worktree(repo: Path, path: Path, branch: str, start: str) -> None:
    """A fresh worktree on a fresh `branch` at `start`, discarding whatever was there."""
    remove_worktree(repo, path)
    delete_branch(repo, branch)
    add_worktree(repo, path, branch, start)


def commit_all(worktree: Path, message: str, paths: list[str] | None = None) -> str | None:
    """Commit every change (or only those under `paths`). Returns the new commit, or None when there is nothing."""
    require_gitfile(worktree)
    git(worktree, "add", "-A", "--", *(paths or ["."]))
    if git_ok(worktree, "diff", "--cached", "--quiet"):
        return None
    git(worktree, "commit", "--no-verify", "-m", message)
    return rev_parse(worktree)


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    """Paths that differ between two commits. A rename counts as its old and its new path."""
    # -z gives the literal names; without it git quotes and escapes non-ASCII paths,
    # and a quoted name would slip past the frozen patterns.
    out = git(repo, "diff", "--name-only", "--no-renames", "-z", base, head)
    return sorted(name for name in out.split("\0") if name)


def diff_text(repo: Path, base: str, head: str) -> str:
    return git(repo, "diff", "--no-renames", base, head)


def clean(worktree: Path) -> list[str]:
    """Make the worktree exactly its commit: drop edits, untracked files and ignored files.

    Returns what was discarded, as `git status` lines.
    """
    require_gitfile(worktree)
    dirty = [line for line in git(worktree, "status", "--porcelain", "--ignored").splitlines() if line]
    if dirty:
        git(worktree, "reset", "--hard", "-q")
        git(worktree, "clean", "-fdxq")
    return dirty


def cherry_pick(worktree: Path, sha: str) -> bool:
    """Apply one commit on top of the worktree's HEAD. On a conflict nothing is left behind and it returns False."""
    require_gitfile(worktree)
    if git_ok(worktree, "cherry-pick", "--allow-empty", sha):
        return True
    git(worktree, "cherry-pick", "--abort", check=False)
    git(worktree, "reset", "--hard", "-q", check=False)
    return False


def set_branch(repo: Path, branch: str, sha: str) -> None:
    """Make `branch` point at `sha`, creating it when needed."""
    git(repo, "branch", "-f", branch, sha)


def fast_forward(worktree: Path, sha: str) -> None:
    require_gitfile(worktree)
    git(worktree, "merge", "--ff-only", sha)


def is_merged(repo: Path, branch: str, into: str) -> bool:
    return git_ok(repo, "merge-base", "--is-ancestor", branch, into)


def remotes(repo: Path) -> list[str]:
    return [line for line in git(repo, "remote").splitlines() if line]
