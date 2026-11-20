import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsi import state


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(self.tmp)}))

    def test_home_follows_the_environment(self):
        self.assertEqual(state.home(), self.tmp)
        with mock.patch.dict(os.environ, {"RSI_HOME": ""}):
            self.assertEqual(state.home(), Path.home() / ".rsi")

    def test_paths(self):
        layout = state.Layout("toy")
        self.assertEqual(layout.root, self.tmp / "toy")
        self.assertEqual(layout.private, self.tmp / "toy" / "private")
        self.assertEqual(layout.split_file, self.tmp / "toy" / "private" / "split.json")
        self.assertEqual(layout.evidence_file, self.tmp / "toy" / "private" / "outcomes" / "evidence.jsonl")
        self.assertTrue(layout.outcomes.is_relative_to(layout.private))
        self.assertEqual(layout.lock_file, self.tmp / "toy" / "lock")

    def test_tasks_path_default_and_override(self):
        layout = state.Layout("toy")
        self.assertEqual(layout.tasks_path(None), layout.private / "tasks.jsonl")
        self.assertEqual(layout.tasks_path("~/bench/t.jsonl"), Path.home() / "bench" / "t.jsonl")

    def test_run_layout_keeps_heldout_private(self):
        run = state.Layout("toy").run("20261120-0900")
        self.assertEqual(run.root, self.tmp / "toy" / "runs" / "20261120-0900")
        self.assertEqual(run.feedback, run.root / "feedback")
        self.assertEqual(run.history, run.root / "history")
        self.assertEqual(run.agents, run.root / "agents")
        private = self.tmp / "toy" / "private" / "runs" / "20261120-0900"
        self.assertEqual(run.private, private)
        self.assertEqual(run.state_file, private / "state.json")
        self.assertEqual(run.heldout, private / "heldout")
        self.assertEqual(state.Layout("toy").private_runs, private.parent)
        self.assertEqual(run.worktrees, self.tmp / "toy" / "worktrees" / "20261120-0900")


class JsonTest(unittest.TestCase):
    def test_write_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "x.json"
            state.write_json_atomic(path, {"a": [1, 2]})
            self.assertEqual(state.read_json(path), {"a": [1, 2]})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["x.json"])

    def test_read_missing_returns_default(self):
        self.assertEqual(state.read_json(Path("/nonexistent/x.json"), {"v": 1}), {"v": 1})


class LockTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "sub" / "lock"

    def test_held_then_released(self):
        with state.Lock(self.path):
            self.assertEqual(self.path.read_text(), str(os.getpid()))
        self.assertFalse(self.path.exists())

    def test_refused_while_a_live_process_holds_it(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(str(os.getppid()))
        with self.assertRaises(state.LockError) as caught:
            with state.Lock(self.path):
                pass
        self.assertIn(str(os.getppid()), str(caught.exception))
        self.assertTrue(self.path.exists())

    def test_stale_lock_is_taken_over(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        self.path.parent.mkdir(parents=True)
        self.path.write_text(str(dead.pid))
        with state.Lock(self.path):
            self.assertEqual(self.path.read_text(), str(os.getpid()))

    def test_unreadable_lock_is_taken_over(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("not a pid")
        with state.Lock(self.path):
            pass


class RunStateTest(unittest.TestCase):
    def test_round_trip(self):
        st = state.RunState(
            run="20261120-0900", project="/p", branch="rsi/20261120-0900", base_ref="main",
            config={"name": "toy"}, benchmark="toy-v1", rounds=2,
            evolve_ids=["t01"], heldout_ids=["t07"],
            start_sha="a" * 40, current_sha="a" * 40, eval_sha="a" * 40,
        )
        self.assertEqual((st.step, st.round, st.status), ("heldout_start", 0, "running"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            st.scores.append(33.33)
            st.save(path)
            loaded = state.RunState.load(path)
            self.assertEqual(loaded, st)
            self.assertEqual(json.loads(path.read_text())["scores"], [33.33])
