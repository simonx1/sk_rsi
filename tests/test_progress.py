import io
import time
import unittest
from unittest import mock

from rsi import progress


class FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FormatTest(unittest.TestCase):
    def test_duration(self):
        self.assertEqual([progress.duration(s) for s in (0, 45, 59.9, 60, 2280, 3600, 15120)],
                         ["0s", "45s", "59s", "1m", "38m", "1h00m", "4h12m"])

    def test_default_template_is_valid(self):
        progress.check_template(progress.DEFAULT_LINE)

    def test_unknown_fields_are_refused(self):
        for line in ("{rnd}", "{metric.Wall}", "{limit.}", "{metric}", "{round.x}", "{}", "{0}"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                progress.check_template(line)
        progress.check_template("{metric.wall_s:>8} {limit.cost_usd} {round!s} plain text {{braces}}")

    def test_format_specs(self):
        snap = progress.Snapshot(run="r1", round=2, score=61.2, metrics={"wall_s": 636.4}, limits={"cost_usd": 6.0},
                                 elapsed=2280, step="measuring")
        self.assertEqual(progress.render("{score:.1f}|{round:d}|{round:>3}|{metric.wall_s:.0f}|{limit.cost_usd:.1f}|"
                                         "{step:>10}|{elapsed:>5}", snap),
                         "61.2|2|  2|636|6.0| measuring|  38m")
        empty = progress.Snapshot(run="r1")
        self.assertEqual(progress.render("{score:.1f} {metric.x:.0f} {eta:>3}", empty), "- -   ?")
        for line in ("{score:.1f}", "{round:d}", "{metric.wall_s:.0f}", "{elapsed:>8}"):
            progress.check_template(line)

    def test_a_spec_that_cannot_render_is_refused(self):
        for line in ("{step:d}", "{round:{nope}}", "{score:zz}", "{run:.2f}"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                progress.check_template(line)

    def test_a_broken_template_is_refused(self):
        with self.assertRaises(ValueError):
            progress.check_template("{round")

    def test_render(self):
        snap = progress.Snapshot(run="r1", status="running", round=2, rounds=5, step="measuring", done=9, total=16,
                                 failed=1, agents_running=0, agents_total=0, errors=3, accepted=1, score=61.2,
                                 start_score=50.0, metrics={"wall_s": 636.0}, limits={"cost_usd": 6.0},
                                 elapsed=15120, step_elapsed=2280, eta=25500)
        line = progress.render("round {round}/{rounds} · {step} · runs {done}/{total}, {failed} failed · agents {agents} · "
                               "score {score} (start {start_score}) · {metric.wall_s} · {limit.cost_usd} · "
                               "{step_elapsed} in step, {elapsed} total, ~{eta} left · {errors} errors, {accepted} kept", snap)
        self.assertEqual(line, "round 2/5 · measuring · runs 9/16, 1 failed · agents - · score 61.20 (start 50.00) · "
                               "636.00 · +6.00% · 38m in step, 4h12m total, ~7h05m left · 3 errors, 1 kept")

    def test_agents_running(self):
        snap = progress.Snapshot(run="r1", agents_running=2, agents_total=3)
        self.assertEqual(progress.render("{agents} {run} {status}", snap), "2/3 r1 running")

    def test_missing_values_render_as_a_dash(self):
        snap = progress.Snapshot(run="r1")
        self.assertEqual(progress.render("{score} {metric.x} {limit.y} {eta} {agents}", snap), "- - - ? -")

    def test_default_line_renders(self):
        line = progress.render(progress.DEFAULT_LINE, progress.Snapshot(run="r1", round=1, rounds=5, step="baseline"))
        self.assertIn("round 1/5 · baseline · runs 0/0, 0 failed", line)


class EtaTest(unittest.TestCase):
    def test_eta(self):
        # runs left x mean run time / parallel, plus agent steps left x mean agent step time
        self.assertEqual(progress.eta_seconds(16, [600, 900], 4, 6, [300, 500]), 16 * 750 / 4 + 6 * 400)
        self.assertEqual(progress.eta_seconds(16, [600], 4, 6, []), 2400)
        self.assertIsNone(progress.eta_seconds(16, [], 4, 6, [600]))
        self.assertEqual(progress.eta_seconds(0, [600], 4, 0, [600]), 0)

    def test_tracker_keeps_each_finished_agent_step(self):
        clock = Clock()
        tracker = progress.Tracker(clock=clock)
        tracker.begin_step("asking agents", 0, agent_step=True)
        tracker.expect(agents=3)
        clock.now = 600
        tracker.begin_step("screening", 0, agent_step=True)
        tracker.expect(agents=1)
        clock.now = 700
        tracker.begin_step("measuring", 0, agent_step=False)
        clock.now = 3000
        tracker.begin_step("selecting", 0, agent_step=True)     # called no agent: not an agent step's time
        clock.now = 3001
        tracker.begin_step("recording", 0, agent_step=False)
        self.assertEqual(tracker.agent_step_seconds, [600, 100])


class PrinterTest(unittest.TestCase):
    def make(self, tty, events="all", width=200, line="{step} {done}/{total}"):
        self.stream, self.clock = FakeStream(tty), Clock()
        self.snap = progress.Snapshot(run="r1", step="measuring", done=1, total=4)
        return progress.Printer(self.stream, line, events, 5, 300, lambda: self.snap, clock=self.clock, width=width)

    def test_terminal_redraws_in_place_and_keeps_permanent_lines(self):
        printer = self.make(tty=True)
        printer.tick()
        self.assertEqual(self.stream.getvalue(), "\r\x1b[Kmeasuring 1/4")
        self.snap.done = 2
        printer.say("[rsi] step started")
        self.assertEqual(self.stream.getvalue(),
                         "\r\x1b[Kmeasuring 1/4\r\x1b[K[rsi] step started\n\r\x1b[Kmeasuring 2/4")

    def test_width_comes_from_the_stream_drawn_on(self):
        import fcntl, os, struct, termios
        main, side = os.openpty()
        self.addCleanup(os.close, main)
        self.addCleanup(os.close, side)
        fcntl.ioctl(side, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 60, 0, 0))

        class PtyStream(FakeStream):
            def fileno(self):
                return side

        stream = PtyStream(True)
        snap = progress.Snapshot(run="r1", step="x" * 200)
        progress.Printer(stream, "{step}", "all", 5, 300, lambda: snap).tick()
        self.assertEqual(len(stream.getvalue()), len(progress.CLEAR) + 59)

    def test_line_is_cut_to_the_terminal_width(self):
        printer = self.make(tty=True, width=5)
        printer.tick()
        self.assertEqual(self.stream.getvalue(), "\r\x1b[Kmeasu")

    def test_log_output_has_no_control_characters(self):
        printer = self.make(tty=False)
        printer.tick()
        self.clock.now = 100
        printer.tick()
        self.clock.now = 301
        printer.tick()
        printer.say("[rsi] step started")
        printer.tick(force=True)
        text = self.stream.getvalue()
        self.assertEqual(text, "[rsi] measuring 1/4\n[rsi] measuring 1/4\n[rsi] step started\n[rsi] measuring 1/4\n")
        self.assertNotIn("\r", text)
        self.assertNotIn("\x1b", text)

    def test_event_levels(self):
        for events, kept in (("all", ["ok", "bad"]), ("failures", ["bad"]), ("none", [])):
            with self.subTest(events=events):
                printer = self.make(tty=False, events=events)
                printer.event("ok", failure=False)
                printer.event("bad", failure=True)
                self.assertEqual(self.stream.getvalue().splitlines(), kept)

    def test_stop_leaves_a_fresh_line(self):
        printer = self.make(tty=True)
        printer.interval_s = 0.01
        printer.start()
        time.sleep(0.05)
        printer.stop()
        self.assertTrue(self.stream.getvalue().endswith("\r\x1b[K"))
        self.assertFalse(printer.running())

    def test_a_bad_snapshot_does_not_stop_the_run(self):
        printer = self.make(tty=False)
        printer.snapshot = lambda: 1 / 0
        printer.tick(force=True)
        self.assertIn("progress unavailable", self.stream.getvalue())

    def test_module_say_without_a_printer(self):
        progress.install(None)
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            progress.say("[rsi] x")
        self.assertEqual(err.getvalue(), "[rsi] x\n")

    def test_module_say_with_a_printer(self):
        printer = self.make(tty=False)
        progress.install(printer)
        try:
            progress.say("[rsi] x")
        finally:
            progress.install(None)
        self.assertEqual(self.stream.getvalue(), "[rsi] x\n")
