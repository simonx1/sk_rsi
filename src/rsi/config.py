"""Load and validate a project's .rsi/config.toml."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    name: str
    base_branch: str
    benchmark: str
    model: str
    editable: tuple[str, ...]
    frozen: tuple[str, ...]
    tasks_file: str | None
    heldout_fraction: float
    rollouts: int
    pass_threshold: float
    heldout: str
    run_task: str
    check: str | None
    ingest: str | None
    rounds: int
    candidates: int
    parallel: int
    task_timeout_s: int
    agent_timeout_s: int
    max_wall_clock_h: float
    tolerance: float
    max_infra_failure_rate: float
    proposer: str | None
    selector: str | None
    pull_request: bool

    def to_dict(self) -> dict:
        data = asdict(self)
        data["editable"] = list(self.editable)
        data["frozen"] = list(self.frozen)
        return data

    @staticmethod
    def from_dict(data: dict) -> "Config":
        values = dict(data)
        values["editable"] = tuple(values["editable"])
        values["frozen"] = tuple(values["frozen"])
        return Config(**values)


_REQUIRED = object()

# (section, key, field, kind, default)
_SCHEMA = (
    ("project", "name", "name", str, _REQUIRED),
    ("project", "base_branch", "base_branch", str, "main"),
    ("setting", "benchmark", "benchmark", str, _REQUIRED),
    ("setting", "model", "model", str, _REQUIRED),
    ("harness", "editable", "editable", tuple, _REQUIRED),
    ("harness", "frozen", "frozen", tuple, []),
    ("tasks", "file", "tasks_file", str, None),
    ("tasks", "heldout_fraction", "heldout_fraction", float, 0.5),
    ("tasks", "rollouts", "rollouts", int, 1),
    ("tasks", "pass_threshold", "pass_threshold", float, 0.5),
    ("tasks", "heldout", "heldout", str, "endpoints"),
    ("commands", "run_task", "run_task", str, _REQUIRED),
    ("commands", "check", "check", str, None),
    ("commands", "ingest", "ingest", str, None),
    ("budget", "rounds", "rounds", int, 5),
    ("budget", "candidates", "candidates", int, 3),
    ("budget", "parallel", "parallel", int, 2),
    ("budget", "task_timeout_s", "task_timeout_s", int, 3600),
    ("budget", "agent_timeout_s", "agent_timeout_s", int, 1800),
    ("budget", "max_wall_clock_h", "max_wall_clock_h", float, 48.0),
    ("budget", "tolerance", "tolerance", float, 1.0),
    ("budget", "max_infra_failure_rate", "max_infra_failure_rate", float, 0.3),
    ("agents", "proposer", "proposer", str, None),
    ("agents", "selector", "selector", str, None),
    ("promotion", "pull_request", "pull_request", bool, True),
)


def _coerce(where: str, raw: object, kind: type) -> object:
    if raw is None:
        return None
    if kind is float:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(raw)
    if kind is int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigError(f"{where} must be a whole number")
        return raw
    if kind is tuple:
        if not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(raw)
    if kind is bool:
        if not isinstance(raw, bool):
            raise ConfigError(f"{where} must be true or false")
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{where} must be a non-empty string")
    return raw


def _check_ranges(cfg: Config) -> None:
    rules = (
        (NAME_RE.match(cfg.name), "project.name may only contain letters, digits, '.', '_' and '-'"),
        (NAME_RE.match(cfg.benchmark), "setting.benchmark may only contain letters, digits, '.', '_' and '-'"),
        (len(cfg.editable) > 0, "harness.editable must list at least one pattern"),
        (0 < cfg.heldout_fraction < 1, "tasks.heldout_fraction must be between 0 and 1, exclusive"),
        (cfg.rollouts >= 1, "tasks.rollouts must be at least 1"),
        (0 <= cfg.pass_threshold <= 1, "tasks.pass_threshold must be from 0 to 1"),
        (cfg.heldout in ("endpoints", "every_round"), 'tasks.heldout must be "endpoints" or "every_round"'),
        (cfg.rounds >= 0, "budget.rounds must be 0 or more"),
        (cfg.candidates >= 1, "budget.candidates must be at least 1"),
        (cfg.parallel >= 1, "budget.parallel must be at least 1"),
        (cfg.task_timeout_s > 0, "budget.task_timeout_s must be positive"),
        (cfg.agent_timeout_s > 0, "budget.agent_timeout_s must be positive"),
        (cfg.max_wall_clock_h > 0, "budget.max_wall_clock_h must be positive"),
        (cfg.tolerance >= 0, "budget.tolerance must be 0 or more"),
        (0 <= cfg.max_infra_failure_rate <= 1, "budget.max_infra_failure_rate must be from 0 to 1"),
    )
    for ok, message in rules:
        if not ok:
            raise ConfigError(message)


def parse(text: str) -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"not valid TOML: {error}") from None
    known: dict[str, set[str]] = {}
    for section, key, *_ in _SCHEMA:
        known.setdefault(section, set()).add(key)
    for section, body in data.items():
        if section not in known:
            raise ConfigError(f"unknown section [{section}]")
        if not isinstance(body, dict):
            raise ConfigError(f"[{section}] must be a table")
        for key in body:
            if key not in known[section]:
                raise ConfigError(f"unknown key {section}.{key}")
    values = {}
    for section, key, field_name, kind, default in _SCHEMA:
        raw = data.get(section, {}).get(key, default)
        if raw is _REQUIRED:
            raise ConfigError(f"{section}.{key} is required")
        values[field_name] = _coerce(f"{section}.{key}", raw, kind)
    cfg = Config(**values)
    _check_ranges(cfg)
    return cfg


def load(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        return parse(path.read_text())
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None
