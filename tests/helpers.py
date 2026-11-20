"""Shared builders for tests."""

import dataclasses
import subprocess
from pathlib import Path

from rsi import config

MINIMAL = """
[project]
name = "toy"

[setting]
benchmark = "toy"
model = "model-a"

[harness]
editable = ["rules.txt"]

[commands]
run_task = "true"
"""


def minimal_config(**overrides) -> config.Config:
    return dataclasses.replace(config.parse(MINIMAL), **overrides)


def git(cwd, *args) -> str:
    done = subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)
    return done.stdout.strip()


def make_repo(path: Path, files: dict[str, str]) -> Path:
    """Create a git repo on branch main with one commit holding `files`."""
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")
    git(path, "config", "commit.gpgsign", "false")
    for name, text in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path.resolve()
