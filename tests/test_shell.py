import time
import unittest
from pathlib import Path

from rsi import shell


class RenderTest(unittest.TestCase):
    def test_substitutes_and_quotes(self):
        cmd = shell.render("run {task} --out {out}", {"task": "a b", "out": "/tmp/x"})
        self.assertEqual(cmd, "run 'a b' --out /tmp/x")

    def test_leaves_unknown_placeholders(self):
        self.assertEqual(shell.render("echo {nope}", {"x": 1}), "echo {nope}")

    def test_empty_value_becomes_an_empty_argument(self):
        self.assertEqual(shell.render("f --since {since}", {"since": ""}), "f --since ''")


class RunTest(unittest.TestCase):
    def test_captures_output_and_code(self):
        r = shell.run("echo out; echo err >&2; exit 3", cwd=Path.cwd(), timeout=10)
        self.assertEqual(
            (r.returncode, r.stdout.strip(), r.stderr.strip(), r.timed_out),
            (3, "out", "err", False),
        )

    def test_env_is_added(self):
        r = shell.run("echo $RSI_X", cwd=Path.cwd(), timeout=10, env={"RSI_X": "hello"})
        self.assertEqual(r.stdout.strip(), "hello")

    def test_timeout_stops_children_too(self):
        started = time.monotonic()
        r = shell.run("sleep 30 & sleep 30", cwd=Path.cwd(), timeout=1)
        self.assertTrue(r.timed_out)
        self.assertLess(time.monotonic() - started, 10)
