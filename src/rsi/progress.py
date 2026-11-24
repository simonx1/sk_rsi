"""The live progress line: what a run is doing, how far it is, and whether anything failed.

Its content is a template with named fields, set under [progress] in a project's config.
"""

from __future__ import annotations

import os
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
    """Refuse a template that names a field the progress line does not have, or that cannot be shown.

    The second check renders the line once with every value missing and once with every value set, so a
    format spec that does not suit its field (`{step:d}`) is refused when the run starts, not hours later.
    """
    try:
        parts = list(string.Formatter().parse(line))
    except ValueError as error:
        raise ValueError(f"progress.line is not a valid template: {error}") from None
    named: dict[str, dict] = {group: {} for group in GROUPS}
    for _, name, _, _ in parts:
        if name is None:
            continue
        group, dot, rest = name.partition(".")
        ok = (not dot and name in FIELDS) or (dot and group in GROUPS and NAME.match(rest))
        if not ok:
            raise ValueError(f"unknown field {name or '{}'} in progress.line")
        if dot:
            named[group][rest] = 1.0
    filled = Snapshot(run="r", round=1, rounds=2, done=1, total=2, failed=1, agents_running=1, agents_total=2,
                      errors=1, accepted=1, score=1.0, start_score=1.0, metrics=named["metric"],
                      limits=named["limit"], elapsed=1.0, step_elapsed=1.0, eta=1.0)
    for snap in (Snapshot(run="r"), filled):
        try:
            render(line, snap)
        except (ValueError, KeyError, IndexError, AttributeError, TypeError) as error:
            raise ValueError(f"progress.line cannot be shown: {error!r}") from None


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


class _Value:
    """One field. With no format spec it shows its usual text; with one, the spec formats the value itself:
    `{score:.1f}` formats the number, `{step:>10}` pads the words. A missing value is `-` either way."""

    def __init__(self, value, text: str | None = None):
        self.value = value
        self.text = ("-" if value is None else str(value)) if text is None else text

    def __format__(self, spec: str) -> str:
        if not spec:
            return self.text
        if self.value is None:
            try:
                return format(self.text, spec)   # keeps a width or an alignment
            except ValueError:
                return self.text                 # a number's spec, such as .1f
        return format(self.value, spec)


class _Group:
    """`metric.<name>` and `limit.<name>`: any name, `-` when there is no value."""

    def __init__(self, values: dict, show):
        self._values, self._show = values, show

    def __getattr__(self, name: str) -> _Value:
        value = self._values.get(name)
        return _Value(value, None if value is None else self._show(value))


def render(line: str, snap: Snapshot) -> str:
    number = lambda value: _Value(value, None if value is None else f"{value:.2f}")
    times = lambda seconds: _Value(duration(seconds))
    values = {
        "run": _Value(snap.run), "status": _Value(snap.status), "step": _Value(snap.step),
        "round": _Value(snap.round), "rounds": _Value(snap.rounds), "done": _Value(snap.done),
        "total": _Value(snap.total), "failed": _Value(snap.failed), "errors": _Value(snap.errors),
        "accepted": _Value(snap.accepted),
        "agents": _Value(f"{snap.agents_running}/{snap.agents_total}" if snap.agents_total else None),
        "score": number(snap.score), "start_score": number(snap.start_score),
        "elapsed": times(snap.elapsed), "step_elapsed": times(snap.step_elapsed),
        "eta": _Value("?") if snap.eta is None else times(snap.eta),
        "metric": _Group(snap.metrics, lambda v: f"{v:.2f}"),
        "limit": _Group(snap.limits, lambda v: f"{v:+.2f}%"),
    }
    return line.format_map(values)


def eta_seconds(runs_left: int, run_seconds: list[float], parallel: int,
                agent_steps_left: int, agent_step_seconds: list[float]) -> float | None:
    """Time left: runs still to make at the average run time so far, divided among the parallel slots,
    plus agent steps still to come at the average time an agent step has taken."""
    if not run_seconds:
        return None
    runs = runs_left * (sum(run_seconds) / len(run_seconds)) / max(parallel, 1)
    agents = agent_steps_left * (sum(agent_step_seconds) / len(agent_step_seconds)) if agent_step_seconds else 0.0
    return runs + agents


CLEAR = "\r\x1b[K"   # back to the start of the line, and erase it


def _columns(stream: TextIO) -> int:
    """The width of the terminal the line is drawn on; stdout may be a file while stderr is the terminal."""
    try:
        return os.get_terminal_size(stream.fileno()).columns
    except (AttributeError, OSError, ValueError):
        return shutil.get_terminal_size().columns


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
        width = self.width or max(_columns(self.stream) - 1, 20)
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
        self.agent_step_seconds: list[float] = []   # each finished step in which agents were called
        self.metrics: dict = {}
        self._agent_step = False
        self._round = 0

    def begin_step(self, word: str, round_no: int, agent_step: bool) -> None:
        with self._lock:
            now = self.clock()
            if self._agent_step and self.agents_total:
                self.agent_step_seconds.append(now - self.step_started)
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

    def agent_steps(self) -> list[float]:
        with self._lock:
            return list(self.agent_step_seconds)
