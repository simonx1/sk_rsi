"""Load and validate a project's .rsi/config.toml."""

from __future__ import annotations

import re
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from . import progress

# Names become directory names, so they start with a letter or digit: ".." would leave the state directory.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


METRIC_RE = re.compile(r"^[a-z][a-z0-9_]*$")
BRANCH_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Constraint:
    """A limit on a metric, relative to the harness a run started from."""
    metric: str
    direction: str   # "increase" or "decrease": the direction the metric may not move too far in
    pct: float       # how far it may move, in percent; 0 means not at all


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
    # Everything below is off by default; a config from before these keys loads unchanged.
    repo: str | None = None              # set only in a config kept outside the repository
    paired: bool = False                 # measure the current harness beside the candidates
    min_gain: float | None = None        # score points a candidate must gain to be accepted
    constraints: tuple[Constraint, ...] = ()
    agent_cli: str = "claude"            # which built-in agent command to use
    agent_model: str | None = None
    agent_effort: str | None = None
    agent_notes: str | None = None      # the owner's notes for the agents; absolute once loaded from a file
    promotion_mode: str = "per_run"      # or "per_edit": one pull request per accepted edit
    draft: bool = False
    branch_prefix: str = "rsi/"
    confirm: bool = False                # measure a chosen candidate once more before keeping it
    progress_line: str = progress.DEFAULT_LINE
    progress_events: str = "all"         # lines for finished task runs and agent calls: "all", "failures", "none"
    progress_interval_s: float = 5.0     # redraw period on a terminal
    progress_log_interval_s: float = 300.0   # period of the plain progress line when the output is not a terminal

    def to_dict(self) -> dict:
        data = asdict(self)
        data["editable"] = list(self.editable)
        data["frozen"] = list(self.frozen)
        data["constraints"] = [asdict(c) for c in self.constraints]
        return data

    @staticmethod
    def from_dict(data: dict) -> "Config":
        values = dict(data)
        values["editable"] = tuple(values["editable"])
        values["frozen"] = tuple(values["frozen"])
        values["constraints"] = tuple(Constraint(**c) for c in values.get("constraints", ()))
        return Config(**values)


_REQUIRED = object()

# (section, key, field, kind, default)
_SCHEMA = (
    ("project", "name", "name", str, _REQUIRED),
    ("project", "base_branch", "base_branch", str, "main"),
    ("project", "repo", "repo", str, None),
    ("setting", "benchmark", "benchmark", str, _REQUIRED),
    ("setting", "model", "model", str, _REQUIRED),
    ("harness", "editable", "editable", tuple, _REQUIRED),
    ("harness", "frozen", "frozen", tuple, []),
    ("tasks", "file", "tasks_file", str, None),
    ("tasks", "heldout_fraction", "heldout_fraction", float, 0.5),
    ("tasks", "rollouts", "rollouts", int, 1),
    ("tasks", "pass_threshold", "pass_threshold", float, 0.5),
    ("tasks", "heldout", "heldout", str, "endpoints"),
    ("tasks", "paired", "paired", bool, False),
    ("tasks", "confirm", "confirm", bool, False),
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
    ("budget", "min_gain", "min_gain", float, None),
    ("agents", "proposer", "proposer", str, None),
    ("agents", "selector", "selector", str, None),
    ("agents", "cli", "agent_cli", str, "claude"),
    ("agents", "model", "agent_model", str, None),
    ("agents", "effort", "agent_effort", str, None),
    ("agents", "notes", "agent_notes", str, None),
    ("promotion", "pull_request", "pull_request", bool, True),
    ("promotion", "mode", "promotion_mode", str, "per_run"),
    ("promotion", "draft", "draft", bool, False),
    ("promotion", "branch_prefix", "branch_prefix", str, "rsi/"),
    ("progress", "line", "progress_line", str, progress.DEFAULT_LINE),
    ("progress", "events", "progress_events", str, "all"),
    ("progress", "interval_s", "progress_interval_s", float, 5.0),
    ("progress", "log_interval_s", "progress_log_interval_s", float, 300.0),
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
    try:
        progress.check_template(cfg.progress_line)
    except ValueError as error:
        raise ConfigError(str(error)) from None
    rules = (
        (NAME_RE.match(cfg.name),
         "project.name must start with a letter or digit and contain only letters, digits, '.', '_' and '-'"),
        (NAME_RE.match(cfg.benchmark),
         "setting.benchmark must start with a letter or digit and contain only letters, digits, '.', '_' and '-'"),
        (cfg.tasks_file is None or cfg.tasks_file.startswith(("/", "~")),
         "tasks.file must be an absolute path or start with ~; a relative path would depend on where rsi is run"),
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
        (cfg.min_gain is None or cfg.min_gain >= 0, "budget.min_gain must be 0 or more"),
        (not cfg.confirm or cfg.paired, "tasks.confirm needs tasks.paired = true"),
        (cfg.progress_events in ("all", "failures", "none"), 'progress.events must be "all", "failures" or "none"'),
        (cfg.progress_interval_s > 0, "progress.interval_s must be positive"),
        (cfg.progress_log_interval_s > 0, "progress.log_interval_s must be positive"),
        (not (cfg.paired and cfg.heldout == "every_round"),
         'tasks.paired cannot be combined with tasks.heldout = "every_round"'),
        (cfg.agent_cli in ("claude", "agy"), 'agents.cli must be "claude" or "agy"'),
        (cfg.agent_cli != "agy" or cfg.agent_model is not None, 'agents.model is required when agents.cli is "agy"'),
        (cfg.promotion_mode in ("per_run", "per_edit"), 'promotion.mode must be "per_run" or "per_edit"'),
        (cfg.promotion_mode != "per_edit" or cfg.repo is not None,
         'promotion.mode = "per_edit" needs a config kept outside the repository (project.repo), '
         "so that each pull request carries only its edit"),
        (BRANCH_PREFIX_RE.match(cfg.branch_prefix),
         "promotion.branch_prefix must start with a letter or digit and contain only letters, digits, '.', '_', '-' and '/'"),
    )
    for ok, message in rules:
        if not ok:
            raise ConfigError(message)


def _constraints(raw: object) -> tuple[Constraint, ...]:
    if not isinstance(raw, list) or not all(isinstance(entry, dict) for entry in raw):
        raise ConfigError("constraints must be a list of tables, each written under [[constraints]]")
    found: list[Constraint] = []
    for index, entry in enumerate(raw):
        where = f"constraints[{index}]"
        for key in entry:
            if key not in ("metric", "max_increase_pct", "max_decrease_pct"):
                raise ConfigError(f"unknown key {where}.{key}")
        if "metric" not in entry:
            raise ConfigError(f"{where}.metric is required")
        metric = entry["metric"]
        if not isinstance(metric, str) or not METRIC_RE.match(metric):
            raise ConfigError(f"{where}.metric must be a metric name such as cost_usd")
        limits = [key for key in ("max_increase_pct", "max_decrease_pct") if key in entry]
        if len(limits) != 1:
            raise ConfigError(f"{where} needs exactly one of max_increase_pct or max_decrease_pct")
        pct = entry[limits[0]]
        if isinstance(pct, bool) or not isinstance(pct, (int, float)) or pct < 0:
            raise ConfigError(f"{where}.{limits[0]} must be a number, 0 or more")
        direction = "increase" if limits[0] == "max_increase_pct" else "decrease"
        if any((c.metric, c.direction) == (metric, direction) for c in found):
            raise ConfigError(f"{where} repeats the limit on {metric}")
        found.append(Constraint(metric, direction, float(pct)))
    return tuple(found)


def parse(text: str) -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"not valid TOML: {error}") from None
    constraints = _constraints(data.pop("constraints", []))
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
    cfg = Config(**values, constraints=constraints)
    _check_ranges(cfg)
    return cfg


def load(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        cfg = parse(path.read_text())
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from None
    if cfg.agent_notes:
        # Relative to the config's own folder, so it means the same file wherever rsi is started from.
        cfg = replace(cfg, agent_notes=str(path.parent.resolve() / Path(cfg.agent_notes).expanduser()))
    return cfg
