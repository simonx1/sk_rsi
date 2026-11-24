"""The live progress line: what a run is doing, how far it is, and whether anything failed.

Its content is a template with named fields, set under [progress] in a project's config.
"""

from __future__ import annotations

import re
import shutil
import string
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO

DEFAULT_LINE = ("round {round}/{rounds} · {step} · runs {done}/{total}, {failed} failed · agents {agents} · "
                "score {score} · {step_elapsed} in step, {elapsed} total, ~{eta} left")

FIELDS = frozenset({"run", "status", "round", "rounds", "step", "done", "total", "failed", "agents", "errors",
                    "accepted", "score", "start_score", "elapsed", "step_elapsed", "eta"})
GROUPS = ("metric", "limit")
NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def check_template(line: str) -> None:
    """Refuse a template that names a field the progress line does not have."""
    try:
        parts = list(string.Formatter().parse(line))
    except ValueError as error:
        raise ValueError(f"progress.line is not a valid template: {error}") from None
    for _, name, _, _ in parts:
        if name is None:
            continue
        group, dot, rest = name.partition(".")
        ok = (not dot and name in FIELDS) or (dot and group in GROUPS and NAME.match(rest))
        if not ok:
            raise ValueError(f"unknown field {name or '{}'} in progress.line")


def duration(seconds: float) -> str:
    """45s, 38m, 4h12m."""
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


@dataclass
class Snapshot:
    """Everything the line can show, at one moment."""
    run: str
    status: str = "running"
    round: int = 0
    rounds: int = 0
    step: str = "starting"
    done: int = 0
    total: int = 0
    failed: int = 0
    agents_running: int = 0
    agents_total: int = 0
    errors: int = 0
    accepted: int = 0
    score: float | None = None
    start_score: float | None = None
    metrics: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    elapsed: float = 0.0
    step_elapsed: float = 0.0
    eta: float | None = None


class _Group:
    """`metric.<name>` and `limit.<name>`: any name, `-` when there is no value."""

    def __init__(self, values: dict, show):
        self._values, self._show = values, show

    def __getattr__(self, name: str) -> str:
        value = self._values.get(name)
        return "-" if value is None else self._show(value)


def render(line: str, snap: Snapshot) -> str:
    number = lambda value: "-" if value is None else f"{value:.2f}"
    values = {
        "run": snap.run, "status": snap.status, "step": snap.step,
        "round": str(snap.round), "rounds": str(snap.rounds), "done": str(snap.done), "total": str(snap.total),
        "failed": str(snap.failed), "errors": str(snap.errors), "accepted": str(snap.accepted),
        "agents": f"{snap.agents_running}/{snap.agents_total}" if snap.agents_total else "-",
        "score": number(snap.score), "start_score": number(snap.start_score),
        "elapsed": duration(snap.elapsed), "step_elapsed": duration(snap.step_elapsed),
        "eta": "?" if snap.eta is None else duration(snap.eta),
        "metric": _Group(snap.metrics, lambda v: f"{v:.2f}"),
        "limit": _Group(snap.limits, lambda v: f"{v:+.2f}%"),
    }
    return line.format_map(values)


def eta_seconds(runs_left: int, run_seconds: list[float], parallel: int,
                agent_rounds_left: int, agent_seconds: list[float]) -> float | None:
    """Time left: runs still to make at the average run time so far, plus agent time per round still to come."""
    if not run_seconds:
        return None
    runs = runs_left * (sum(run_seconds) / len(run_seconds)) / max(parallel, 1)
    agents = agent_rounds_left * (sum(agent_seconds) / len(agent_seconds)) if agent_seconds else 0.0
    return runs + agents


CLEAR = "\r\x1b[K"   # back to the start of the line, and erase it


class Printer:
    """Prints the live line and the permanent lines above it.

    On a terminal the live line is rewritten in place every `interval_s`; otherwise (a log file, nohup)
    it is printed as an ordinary line every `log_interval_s`, so a log carries no control characters.
    """

    def __init__(self, stream: TextIO, line: str, events: str, interval_s: float, log_interval_s: float,
                 snapshot: Callable[[], Snapshot], clock: Callable[[], float] = time.monotonic,
                 width: int | None = None):
        self.stream, self.line, self.events = stream, line, events
        self.interval_s, self.log_interval_s = interval_s, log_interval_s
        self.snapshot, self.clock, self.width = snapshot, clock, width
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_log: float | None = None

    def _text(self) -> str:
        try:
            return render(self.line, self.snapshot())
        except Exception as error:  # the progress line must never stop a run
            return f"progress unavailable: {error}"

    def _draw(self) -> None:
        width = self.width or max(shutil.get_terminal_size().columns - 1, 20)
        self.stream.write(CLEAR + self._text()[:width])
        self.stream.flush()

    def tick(self, force: bool = False) -> None:
        with self._lock:
            if self.tty:
                self._draw()
                return
            now = self.clock()
            if force or self._last_log is None or now - self._last_log >= self.log_interval_s:
                self.stream.write(f"[rsi] {self._text()}\n")
                self.stream.flush()
                self._last_log = now

    def say(self, text: str) -> None:
        """A line that stays."""
        with self._lock:
            if self.tty:
                self.stream.write(CLEAR + text + "\n")
                self._draw()
            else:
                self.stream.write(text + "\n")
                self.stream.flush()

    def event(self, text: str, failure: bool) -> None:
        """A finished task run or agent call, kept or dropped by the `events` setting."""
        if self.events == "all" or (self.events == "failures" and failure):
            self.say(text)

    def start(self) -> None:
        self._stop.clear()
        self.tick()

        def loop() -> None:
            while not self._stop.wait(self.interval_s):
                self.tick()

        self._thread = threading.Thread(target=loop, name="rsi-progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            if self.tty:
                self.stream.write(CLEAR)  # leave the cursor on an empty line
                self.stream.flush()

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


_installed: Printer | None = None


def install(printer: Printer | None) -> None:
    global _installed
    _installed = printer


def say(text: str) -> None:
    """Print a line that stays: above the live line when one is shown, else plainly to stderr."""
    if _installed is not None:
        _installed.say(text)
    else:
        print(text, file=sys.stderr, flush=True)


def refresh() -> None:
    """Show the line now: printed at once in a log, so even a short step leaves one line."""
    if _installed is not None:
        _installed.tick(force=True)


def event(text: str, failure: bool) -> None:
    if _installed is not None:
        _installed.event(text, failure)
    elif failure:
        print(text, file=sys.stderr, flush=True)


class Tracker:
    """The counters behind the line: the current step's task runs and agent calls, and timings."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self.step = "starting"
        self.step_started = clock()
        self.done = self.total = self.failed = 0
        self.agents_running = self.agents_total = 0
        self.errors = 0
        self.run_seconds: list[float] = []
        self.agent_seconds: dict[int, float] = {}   # per round: time spent in steps where agents work
        self.metrics: dict = {}
        self._agent_step = False
        self._round = 0

    def begin_step(self, word: str, round_no: int, agent_step: bool) -> None:
        with self._lock:
            now = self.clock()
            if self._agent_step:
                self.agent_seconds[self._round] = self.agent_seconds.get(self._round, 0.0) + now - self.step_started
            self.step, self._round, self._agent_step, self.step_started = word, round_no, agent_step, now
            self.done = self.total = self.failed = 0
            self.agents_running = self.agents_total = 0

    def expect(self, runs: int = 0, agents: int = 0) -> None:
        with self._lock:
            self.total += runs
            self.agents_total += agents

    def run_finished(self, error: bool, cached: bool, seconds: float) -> None:
        with self._lock:
            self.done += 1
            if error:
                self.failed += 1
                self.errors += 1
            if not cached and seconds:
                self.run_seconds.append(seconds)

    def agent_started(self) -> None:
        with self._lock:
            self.agents_running += 1

    def agent_finished(self, ok: bool) -> None:
        with self._lock:
            self.agents_running = max(self.agents_running - 1, 0)
            if not ok:
                self.errors += 1

    def finished_agent_rounds(self) -> list[float]:
        with self._lock:
            return [seconds for round_no, seconds in self.agent_seconds.items() if round_no < self._round]
