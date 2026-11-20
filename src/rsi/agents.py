"""Run a proposer or selector as a headless command and read back its JSON output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import shell
from .state import write_json_atomic

# The default agent has file tools only. Without a shell it cannot read around the deny
# rules in {settings_file}, which is what keeps held-out data away from it.
DEFAULT_COMMAND = (
    "claude -p --permission-mode acceptEdits "
    "--allowedTools Read Edit Write Grep Glob "
    "--disallowedTools Bash "
    "--add-dir {feedback_dir} {history_dir} {out_dir} "
    "--settings {settings_file} < {prompt_file}"
)


class AgentOutputError(Exception):
    pass


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    error: str | None


def _write_settings(path: Path, private_dir: Path) -> None:
    root = str(private_dir.resolve()).lstrip("/")
    deny = [f"{tool}(//{root}/**)" for tool in ("Read", "Edit", "Write")]
    write_json_atomic(path, {"permissions": {"deny": deny}})


def run_agent(
    role: str,
    command: str | None,
    prompt: str,
    cwd: Path,
    out_dir: Path,
    feedback_dir: Path,
    history_dir: Path,
    private_dir: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> AgentResult:
    """Run one agent call. Its answer is whatever JSON file it leaves in out_dir."""
    for directory in (out_dir, feedback_dir, history_dir):
        directory.mkdir(parents=True, exist_ok=True)
    prompt_file = out_dir / "prompt.md"
    prompt_file.write_text(prompt)
    settings_file = out_dir / "settings.json"
    _write_settings(settings_file, private_dir)
    rendered = shell.render(command or DEFAULT_COMMAND, {
        "prompt_file": prompt_file,
        "out_dir": out_dir,
        "feedback_dir": feedback_dir,
        "history_dir": history_dir,
        "settings_file": settings_file,
    })
    done = shell.run(rendered, cwd=cwd, timeout=timeout, env={"RSI_ROLE": role, **(env or {})})
    (out_dir / "agent.log").write_text(done.stdout + "\n--- stderr ---\n" + done.stderr)
    if done.timed_out:
        return AgentResult(False, f"timed out after {timeout}s")
    if done.returncode != 0:
        return AgentResult(False, f"exit code {done.returncode}")
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
    return {
        "root_cause": {key: value.strip()},
        "what": what.strip(),
        "predicted": list(dict.fromkeys(predicted)),
        "risk": risk,
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
        if not isinstance(row, dict) or row.get("sample") not in verdicts:
            continue
        if not isinstance(row.get("keep"), bool):
            raise AgentOutputError("each screen entry needs keep: true or false")
        reason = row.get("reason", "")
        verdicts[row["sample"]] = (row["keep"], reason if isinstance(reason, str) else "")
    return verdicts


def read_decision(out_dir: Path, samples: list[str]) -> tuple[str | None, str]:
    obj = _load(out_dir, "decision.json")
    if "accept" not in obj:
        raise AgentOutputError("decision.json needs an 'accept' key (a sample name or null)")
    accept = obj["accept"]
    if accept is not None and accept not in samples:
        raise AgentOutputError(f"decision.json accepts {accept!r}, which is not an evaluated candidate")
    reason = obj.get("reason", "")
    return accept, reason if isinstance(reason, str) else ""
