"""Run a proposer or selector as a headless command and read back its JSON output."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path

from . import shell
from .state import write_json_atomic

# The default agent is confined three ways:
# - --restricted limits the file tools to the working directory and the --add-dir folders,
#   removes the tools that run commands, and ignores the user's own settings;
# - --tools fixes which tools exist, and --strict-mcp-config loads no MCP servers;
# - {settings_file} denies the places held-out data is kept, and makes the shared
#   feedback and history folders read-only.
DEFAULT_COMMAND = (
    "claude -p --restricted --strict-mcp-config --permission-mode acceptEdits "
    "--tools Read,Edit,Write,Grep,Glob "
    "--disallowedTools Bash "
    "--add-dir {feedback_dir} {history_dir} {out_dir} "
    "--settings {settings_file} < {prompt_file}"
)


# agy takes its prompt as an argument, so it is given this short text, which names the prompt file.
# Headless agy ends its session at once, with no reply, when it is denied an action: any shell
# command, and any file outside its working directory and the added folders. It has no per-path
# deny rules; what keeps it from changing the shared folders is read_only() below.
INSTRUCTION = (
    "Read the file {prompt_file} and follow it exactly. You are running without a person: use only your file\n"
    "viewing and file editing tools. Never run a shell command, and never open a file outside the current\n"
    "directory and the folders that file names. Either one ends this session at once with nothing saved.\n"
)

AGY_COMMAND = (
    "agy --model {model}{effort} --mode accept-edits "
    "--add-dir {{feedback_dir}} --add-dir {{history_dir}} --add-dir {{out_dir}} "
    "--print-timeout {timeout}s --output-format json "
    '-p "$(cat {{instruction_file}})" > {{out_dir}}/agent-result.json'
)


def builtin_command(cli: str, model: str | None, effort: str | None, timeout_s: int) -> str:
    """The command template for a built-in agent. `model` and `effort` are shell-quoted here."""
    if cli == "agy":
        return AGY_COMMAND.format(
            model=shlex.quote(model or ""),
            effort=f" --effort {shlex.quote(effort)}" if effort else "",
            timeout=int(timeout_s),
        )
    if model:
        return DEFAULT_COMMAND.replace("claude -p ", f"claude -p --model {shlex.quote(model)} ", 1)
    return DEFAULT_COMMAND


METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class AgentOutputError(Exception):
    pass


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    error: str | None
    timed_out: bool = False
    denied: bool = False   # the agent's own report says an action was denied; what it wrote before may still be valid


_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def set_writable(paths: list[Path], writable: bool) -> None:
    """Make every file and folder under `paths` read-only on disk, or writable by the owner again."""
    for top in paths:
        if not top.exists():
            continue
        entries = [top]
        for folder, folders, files in os.walk(top):
            entries += [Path(folder) / name for name in folders + files]
        for entry in entries:
            if entry.is_symlink():
                continue
            mode = entry.stat().st_mode
            entry.chmod(mode | stat.S_IWUSR if writable else mode & ~_WRITE_BITS)


@contextlib.contextmanager
def read_only(paths: list[Path]):
    """While agents run, the folders they are given to read cannot be changed, whatever agent it is."""
    set_writable(paths, False)
    try:
        yield
    finally:
        set_writable(paths, True)


def _denied_actions(out_dir: Path) -> list[str]:
    """Actions the agent's own result file says were denied (agy writes one; other agents do not)."""
    try:
        result = json.loads((out_dir / "agent-result.json").read_text())
    except (OSError, ValueError):
        return []
    denied = result.get("denied_actions") if isinstance(result, dict) else None
    if not isinstance(denied, list):
        return []
    names = [str(d.get("action", "unknown")) for d in denied if isinstance(d, dict)]
    return list(dict.fromkeys(names))


def _targets(path: Path) -> list[str]:
    """Permission-rule targets for a path: a directory with everything under it, or one file."""
    suffix = "" if path.is_file() else "/**"
    # A path reached through a symlink has two names; cover both.
    names = dict.fromkeys((path.expanduser().absolute(), path.resolve()))
    return ["//" + str(name).lstrip("/") + suffix for name in names]


def _write_settings(path: Path, deny: list[Path], readonly: list[Path]) -> None:
    """Deny reading and editing `deny`, and editing `readonly`.

    claude applies Read and Edit rules to file access (Edit covers every editing tool, Read
    covers Grep and Glob); it ignores Write rules.
    """
    rules = []
    for denied in deny:
        for target in _targets(denied):
            rules += [f"Read({target})", f"Edit({target})"]
    for kept in readonly:
        rules += [f"Edit({target})" for target in _targets(kept)]
    write_json_atomic(path, {"permissions": {"deny": rules}})


def run_agent(
    role: str,
    command: str | None,
    prompt: str,
    cwd: Path,
    out_dir: Path,
    feedback_dir: Path,
    history_dir: Path,
    deny: list[Path],
    timeout: float,
    env: dict[str, str] | None = None,
    readonly: list[Path] | tuple[Path, ...] = (),
) -> AgentResult:
    """Run one agent call. Its answer is whatever JSON file it leaves in out_dir.

    `deny` paths may be neither read nor edited; `readonly` paths may be read only.
    """
    for directory in (out_dir, feedback_dir, history_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prompt_file = out_dir / "prompt.md"
    prompt_file.write_text(prompt)
    settings_file = out_dir / "settings.json"
    _write_settings(settings_file, deny, list(readonly))
    instruction_file = out_dir / "instruction.txt"
    instruction_file.write_text(INSTRUCTION.replace("{prompt_file}", str(prompt_file)))
    rendered = shell.render(command or DEFAULT_COMMAND, {
        "prompt_file": prompt_file,
        "instruction_file": instruction_file,
        "out_dir": out_dir,
        "feedback_dir": feedback_dir,
        "history_dir": history_dir,
        "settings_file": settings_file,
    })
    done = shell.run(rendered, cwd=cwd, timeout=timeout, env={"RSI_ROLE": role, **(env or {})})
    (out_dir / "agent.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    if done.timed_out:
        return AgentResult(False, f"timed out after {timeout}s", timed_out=True)
    if done.returncode != 0:
        return AgentResult(False, f"exit code {done.returncode}")
    denied = _denied_actions(out_dir)
    if denied:
        return AgentResult(False, "the agent was denied: " + ", ".join(denied), denied=True)
    return AgentResult(True, None)


def _load(out_dir: Path, name: str) -> dict:
    path = out_dir / name
    if not path.is_file():
        raise AgentOutputError(f"{name} was not written")
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise AgentOutputError(f"{name} is not valid JSON") from None
    if not isinstance(obj, dict):
        raise AgentOutputError(f"{name} must be a JSON object")
    return obj


def validate_manifest(obj: object) -> dict:
    if not isinstance(obj, dict):
        raise AgentOutputError("manifest.json must be a JSON object")
    cause = obj.get("root_cause")
    if not isinstance(cause, dict) or len(cause) != 1:
        raise AgentOutputError('root_cause must be {"id": "..."} or {"new": "..."}')
    key, value = next(iter(cause.items()))
    if key not in ("id", "new") or not isinstance(value, str) or not value.strip():
        raise AgentOutputError('root_cause must be {"id": "..."} or {"new": "..."} with non-empty text')
    what = obj.get("what")
    if not isinstance(what, str) or not what.strip():
        raise AgentOutputError("what must be non-empty text")
    predicted = obj.get("predicted", [])
    if not isinstance(predicted, list) or not all(isinstance(t, str) for t in predicted):
        raise AgentOutputError("predicted must be a list of task ids")
    risk = obj.get("risk", "")
    if not isinstance(risk, str):
        raise AgentOutputError("risk must be text")
    title = obj.get("title")
    if title is not None:
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 72:
            raise AgentOutputError("title must be text of at most 72 characters")
        title = title.strip()
    expected = obj.get("expected", {})
    valid = isinstance(expected, dict) and all(
        isinstance(name, str) and METRIC_NAME.match(name)
        and isinstance(change, (int, float)) and not isinstance(change, bool) and math.isfinite(change)
        for name, change in expected.items())
    if not valid:
        raise AgentOutputError("expected must map metric names to the percent change you expect, as numbers")
    return {
        "root_cause": {key: value.strip()},
        "what": what.strip(),
        "predicted": list(dict.fromkeys(predicted)),
        "risk": risk,
        "title": title,
        "expected": {name: float(change) for name, change in expected.items()},
    }


def read_manifest(out_dir: Path) -> dict:
    return validate_manifest(_load(out_dir, "manifest.json"))


def read_screen(out_dir: Path, samples: list[str]) -> dict[str, tuple[bool, str]]:
    """The selector's keep/drop verdict per sample. A sample it did not mention is kept."""
    rows = _load(out_dir, "screen.json").get("screen")
    if not isinstance(rows, list):
        raise AgentOutputError("screen.json needs a 'screen' list")
    verdicts = {sample: (True, "not mentioned by the selector") for sample in samples}
    for row in rows:
        sample = row.get("sample") if isinstance(row, dict) else None
        if not isinstance(sample, str) or sample not in verdicts:
            continue
        if not isinstance(row.get("keep"), bool):
            raise AgentOutputError("each screen entry needs keep: true or false")
        reason = row.get("reason", "")
        verdicts[row["sample"]] = (row["keep"], reason if isinstance(reason, str) else "")
    return verdicts


def read_decision(out_dir: Path, samples: list[str]) -> tuple[str | None, str, list[str]]:
    """The selector's choice, its reason, and the further candidates it would also keep, best first."""
    obj = _load(out_dir, "decision.json")
    if "accept" not in obj:
        raise AgentOutputError("decision.json needs an 'accept' key (a sample name or null)")
    accept = obj["accept"]
    if accept is not None and accept not in samples:
        raise AgentOutputError(f"decision.json accepts {accept!r}, which is not an evaluated candidate")
    reason = obj.get("reason", "")
    # `then` is optional, so a bad one is cut down to what is usable rather than refused: refusing it
    # would throw away the round's measurements for a field the round can do without.
    given = obj.get("then")
    then: list[str] = []
    if accept is not None and isinstance(given, list):
        for sample in given:
            if isinstance(sample, str) and sample in samples and sample != accept and sample not in then:
                then.append(sample)
    return accept, reason if isinstance(reason, str) else "", then
