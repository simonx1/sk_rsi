"""Where a project's loop state lives on disk, and how it is saved."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("RSI_HOME") or Path.home() / ".rsi")


@dataclass(frozen=True)
class RunLayout:
    root: Path       # runs/<run>: feedback, history and agent folders, which agents may read
    private: Path    # private/runs/<run>: the run state and held-out results, never shown to agents
    worktrees: Path  # worktrees/<run>

    @property
    def state_file(self) -> Path:
        return self.private / "state.json"

    @property
    def heldout(self) -> Path:
        return self.private / "heldout"

    @property
    def feedback(self) -> Path:
        return self.root / "feedback"

    @property
    def history(self) -> Path:
        return self.root / "history"

    @property
    def agents(self) -> Path:
        return self.root / "agents"


@dataclass(frozen=True)
class Layout:
    name: str

    @property
    def root(self) -> Path:
        return home() / self.name

    @property
    def private(self) -> Path:
        return self.root / "private"

    @property
    def split_file(self) -> Path:
        return self.private / "split.json"

    @property
    def pending_tasks(self) -> Path:
        return self.private / "pending_tasks.jsonl"

    @property
    def benchmark_file(self) -> Path:
        return self.private / "benchmark.json"

    @property
    def outcomes(self) -> Path:
        # Under private/: stored evidence can be about held-out tasks.
        return self.private / "outcomes"

    @property
    def evidence_file(self) -> Path:
        return self.outcomes / "evidence.jsonl"

    @property
    def evidence_cursor(self) -> Path:
        return self.outcomes / "cursor.json"

    @property
    def seen_ids(self) -> Path:
        return self.outcomes / "seen_ids.txt"

    @property
    def ingest_stamp(self) -> Path:
        return self.outcomes / "last_ingest"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def private_runs(self) -> Path:
        return self.private / "runs"

    @property
    def lock_file(self) -> Path:
        return self.root / "lock"

    def tasks_path(self, tasks_file: str | None) -> Path:
        if tasks_file:
            return Path(tasks_file).expanduser()
        return self.private / "tasks.jsonl"

    def run(self, run_id: str) -> RunLayout:
        return RunLayout(
            root=self.runs_dir / run_id,
            private=self.private_runs / run_id,
            worktrees=self.root / "worktrees" / run_id,
        )


def write_json_atomic(path: Path, obj: object) -> None:
    """Write JSON so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def read_json(path: Path, default: object = None) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


class LockError(Exception):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """One loop per project. A lock left by a dead process is taken over."""

    def __init__(self, path: Path):
        self.path = path

    def _holder(self) -> int | None:
        try:
            return int(self.path.read_text().strip())
        except (ValueError, FileNotFoundError):
            return None

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mine = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        mine.write_text(str(os.getpid()))
        try:
            for _ in range(2):
                try:
                    os.link(mine, self.path)  # atomic: the lock appears complete, or the call fails
                    return self
                except FileExistsError:
                    pid = self._holder()
                    if pid is not None and pid != os.getpid() and _alive(pid):
                        raise LockError(f"another rsi run (pid {pid}) holds {self.path}") from None
                    self.path.unlink(missing_ok=True)  # left by a dead process, or unreadable
            raise LockError(f"could not take {self.path}; another rsi run took it first")
        finally:
            mine.unlink(missing_ok=True)

    def __exit__(self, *exc: object) -> None:
        if self._holder() == os.getpid():
            self.path.unlink(missing_ok=True)


@dataclass
class RunState:
    run: str
    project: str
    branch: str
    base_ref: str
    config: dict
    benchmark: str            # label with version, e.g. "replay-v1"
    rounds: int
    evolve_ids: list[str]
    heldout_ids: list[str]
    start_sha: str
    current_sha: str          # tip of the run branch
    eval_sha: str             # evaluated commit the current harness corresponds to
    step: str = "heldout_start"
    round: int = 0
    status: str = "running"   # running | error | done | abandoned
    error: str | None = None
    scores: list[float] = field(default_factory=list)   # evolve: start, then one per finished round
    heldout: dict = field(default_factory=dict)         # {"start": x, "end": y, "rounds": {"0": z}}
    candidates: list[dict] = field(default_factory=list)
    decision: dict | None = None
    rounds_log: list[dict] = field(default_factory=list)
    active_seconds: float = 0.0
    promoted: dict | None = None

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))

    @staticmethod
    def load(path: Path) -> "RunState":
        return RunState(**json.loads(path.read_text()))
