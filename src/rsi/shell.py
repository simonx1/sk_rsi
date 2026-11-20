"""Run a shell command with a timeout that also stops its children."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

_guard = threading.Lock()
_live: set[int] = set()     # process groups of commands now running
_stopped: set[int] = set()  # those kill_all() stopped


class Interrupted(Exception):
    """The command was stopped by kill_all(), so it has no result."""


@dataclass(frozen=True)
class ShellResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def render(template: str, values: dict[str, object]) -> str:
    """Replace each {name} with its shell-quoted value. Unknown names are left alone."""
    out = template
    for name, value in values.items():
        out = out.replace("{" + name + "}", shlex.quote(str(value)))
    return out


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def kill_all() -> None:
    """Stop every running command. Each one's run() call then raises Interrupted.

    Commands run in their own sessions, so Ctrl-C in the terminal does not reach them by itself.
    """
    with _guard:
        pids = list(_live)
        _stopped.update(pids)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(command: str, cwd: Path, timeout: float, env: dict[str, str] | None = None) -> ShellResult:
    """Run `command` through sh in its own process group, so a timeout stops everything it started."""
    full_env = dict(os.environ)
    full_env.update(env or {})
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    with _guard:
        _live.add(proc.pid)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
            result = ShellResult(proc.returncode, out, err, False)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, err = proc.communicate()
            result = ShellResult(-1, out or "", err or "", True)
        except BaseException:
            _kill_group(proc)
            proc.wait()
            raise
    finally:
        with _guard:
            _live.discard(proc.pid)
            stopped = proc.pid in _stopped
            _stopped.discard(proc.pid)
    if stopped:
        raise Interrupted(command)
    return result
