import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsi import tasks
from rsi.state import Layout


def write_tasks(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "text": i}) + "\n" for i in ids))


class LoadTasksTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "tasks.jsonl"

    def assert_error(self, text, fragment):
        self.path.write_text(text)
        with self.assertRaises(tasks.TaskError) as caught:
            tasks.load_tasks(self.path)
        self.assertIn(fragment, str(caught.exception))

    def test_loads_and_skips_blank_lines(self):
        self.path.write_text('{"id": "a", "x": 1}\n\n{"id": "b.2_c-d"}\n')
        self.assertEqual([t["id"] for t in tasks.load_tasks(self.path)], ["a", "b.2_c-d"])

    def test_missing_file(self):
        with self.assertRaises(tasks.TaskError) as caught:
            tasks.load_tasks(self.path)
        self.assertIn("not found", str(caught.exception))

    def test_bad_lines(self):
        self.assert_error("nope\n", ":1: not JSON")
        self.assert_error("[1]\n", ":1: must be a JSON object")
        self.assert_error('{"id": "a"}\n{"text": "x"}\n', ":2: id must match")
        self.assert_error('{"id": "a b"}\n', ":1: id must match")
        for unsafe in ("..", ".", ".hidden", "-flag"):
            self.assert_error(json.dumps({"id": unsafe}) + "\n", ":1: id must match")
        self.assert_error('{"id": "a"}\n{"id": "a"}\n', ":2: duplicate id a")


class AssignSplitTest(unittest.TestCase):
    def test_balanced_halves(self):
        ids = [f"t{i:02d}" for i in range(12)]
        split = tasks.assign_split(ids, {}, 0.5)
        self.assertEqual(sorted(split), ids)
        self.assertEqual(sum(1 for s in split.values() if s == "heldout"), 6)

    def test_deterministic_regardless_of_input_order(self):
        ids = [f"t{i:02d}" for i in range(9)]
        self.assertEqual(tasks.assign_split(ids, {}, 0.5), tasks.assign_split(list(reversed(ids)), {}, 0.5))

    def test_existing_assignments_never_change(self):
        ids = [f"t{i:02d}" for i in range(6)]
        first = tasks.assign_split(ids, {}, 0.5)
        more = ids + [f"n{i}" for i in range(6)]
        second = tasks.assign_split(more, first, 0.5)
        for task_id in ids:
            self.assertEqual(second[task_id], first[task_id])
        self.assertEqual(sum(1 for s in second.values() if s == "heldout"), 6)

    def test_new_tasks_restore_the_balance(self):
        existing = {"a": "evolve", "b": "evolve", "c": "evolve"}
        split = tasks.assign_split(["a", "b", "c", "d", "e", "f"], existing, 0.5)
        self.assertEqual([split[t] for t in ("d", "e", "f")], ["heldout"] * 3)

    def test_fraction_is_respected(self):
        ids = [f"t{i:02d}" for i in range(20)]
        split = tasks.assign_split(ids, {}, 0.25)
        self.assertEqual(sum(1 for s in split.values() if s == "heldout"), 5)

    def test_first_task_goes_to_evolve(self):
        self.assertEqual(tasks.assign_split(["only"], {}, 0.5), {"only": "evolve"})


class StateFilesTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(tmp)}))
        self.layout = Layout("toy")
        self.tasks_path = self.layout.tasks_path(None)

    def test_split_is_saved_and_reused(self):
        first = tasks.load_split(self.layout, ["a", "b", "c", "d"], 0.5)
        self.assertEqual(json.loads(self.layout.split_file.read_text()), first)
        self.assertEqual(tasks.load_split(self.layout, ["d", "c", "b", "a"], 0.5), first)

    def test_a_saved_split_wins_over_what_would_be_computed(self):
        self.layout.private.mkdir(parents=True)
        self.layout.split_file.write_text(json.dumps({"a": "heldout", "b": "heldout", "c": "heldout"}))
        split = tasks.load_split(self.layout, ["a", "b", "c", "d"], 0.5)
        self.assertEqual(split, {"a": "heldout", "b": "heldout", "c": "heldout", "d": "evolve"})

    def test_benchmark_version_starts_at_one(self):
        self.assertEqual(tasks.benchmark_version(self.layout), 1)
        self.assertEqual(tasks.benchmark_label("replay", self.layout), "replay-v1")

    def test_pending_tasks_are_deduplicated(self):
        added = tasks.add_pending(self.layout, [{"id": "p1"}, {"id": "p2"}, {"id": "known"}], {"known"})
        self.assertEqual(added, 2)
        self.assertEqual(tasks.add_pending(self.layout, [{"id": "p2"}, {"id": "p3"}], set()), 1)
        self.assertEqual(tasks.pending_count(self.layout), 3)

    def test_fold_moves_pending_into_the_benchmark_and_bumps_the_version(self):
        write_tasks(self.tasks_path, ["a", "b"])
        tasks.add_pending(self.layout, [{"id": "p1"}, {"id": "a"}], set())
        self.assertEqual(tasks.fold_pending(self.layout, self.tasks_path), 1)
        self.assertEqual([t["id"] for t in tasks.load_tasks(self.tasks_path)], ["a", "b", "p1"])
        self.assertEqual(tasks.pending_count(self.layout), 0)
        self.assertEqual(tasks.benchmark_label("replay", self.layout), "replay-v2")

    def test_fold_into_a_task_file_without_a_final_newline(self):
        self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks_path.write_text('{"id": "a"}')
        tasks.add_pending(self.layout, [{"id": "p1"}], set())
        self.assertEqual(tasks.fold_pending(self.layout, self.tasks_path), 1)
        self.assertEqual([t["id"] for t in tasks.load_tasks(self.tasks_path)], ["a", "p1"])

    def test_fold_with_nothing_pending_changes_nothing(self):
        write_tasks(self.tasks_path, ["a"])
        self.assertEqual(tasks.fold_pending(self.layout, self.tasks_path), 0)
        self.assertEqual(tasks.benchmark_version(self.layout), 1)
