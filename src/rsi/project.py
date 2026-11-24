"""Find a repository's RSI config: inside the repository, or under RSI_HOME for a project kept outside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config, state, workspace
from .config import Config, ConfigError
from .state import Layout

CONFIG_PATH = ".rsi/config.toml"
SIM_PATH = ".rsi/sim.json"
LOG_PATH = ".rsi/log.md"


@dataclass(frozen=True)
class Project:
    repo: Path          # the repository's root
    cfg: Config         # the config as it is on disk now
    layout: Layout
    external: bool      # True when the config, memory and log live under RSI_HOME, not in the repository
    config_path: Path

    @property
    def sim_path(self) -> Path:
        """The memory file outside a run: the working copy's, or the one under RSI_HOME."""
        return self.layout.sim_file if self.external else self.repo / SIM_PATH


def _external_configs(repo: Path) -> list[tuple[Path, Config]]:
    """Every readable config under RSI_HOME whose project.repo is this repository."""
    home = state.home()
    if not home.is_dir():
        return []
    found = []
    for path in sorted(home.glob("*/config.toml")):
        try:
            cfg = config.load(path)
        except ConfigError:
            continue  # another project's broken config is not this project's problem
        if cfg.repo and Path(cfg.repo).expanduser().resolve() == repo.resolve():
            found.append((path, cfg))
    return found


def find(path: Path) -> Project:
    """The project that `path` belongs to. Raises ConfigError when it has no config, or an ambiguous one."""
    repo = workspace.repo_root(path)
    inside = repo / CONFIG_PATH
    if inside.is_file():
        cfg = config.load(inside)
        if cfg.repo is not None:
            raise ConfigError(f"{inside}: project.repo is only for a config kept outside the repository")
        return Project(repo, cfg, Layout(cfg.name), False, inside)
    matches = _external_configs(repo)
    if len(matches) > 1:
        raise ConfigError("more than one config names this repository: " + ", ".join(str(p) for p, _ in matches))
    if not matches:
        raise ConfigError(
            f"no RSI config for {repo}: expected {inside}, or a config under {state.home()} "
            "whose project.repo is this repository")
    config_path, cfg = matches[0]
    if config_path.parent.name != cfg.name:
        raise ConfigError(
            f"{config_path}: project.name is {cfg.name!r}, but the config is in the folder "
            f"{config_path.parent.name!r}; they must be the same")
    return Project(repo, cfg, Layout(cfg.name), True, config_path)
