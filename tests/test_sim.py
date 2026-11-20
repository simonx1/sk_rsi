import tempfile
import unittest
from pathlib import Path

from rsi import sim


def attempt(run="r1", round_no=0, **over):
    base = {
        "run": run, "round": round_no, "sample": "s1", "accepted": True, "backfilled": False,
        "what": "edit", "files": ["a.txt"], "predicted": [], "fixed": [], "missed": [],
        "regressed": [], "passed": 1, "total": 2, "score_before": 0.0, "score_after": 50.0,
        "domain": {"benchmark": "b-v1", "model": "m"},
    }
    base.update(over)
    return base


class FileTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(self.enterContext(tempfile.TemporaryDirectory())) / ".rsi" / "sim.json"

    def test_missing_file_is_an_empty_memory(self):
        self.assertEqual(sim.load(self.path), {"next_id": 1, "carried_from": [], "aliases": {}, "problems": []})

    def test_round_trip(self):
        memory = sim.empty()
        sim.record_attempt(memory, {"new": "premature completion"}, attempt())
        sim.save(self.path, memory)
        self.assertEqual(sim.load(self.path), memory)

    def test_corrupt_file(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{nope")
        with self.assertRaises(sim.SimError):
            sim.load(self.path)
        self.path.write_text('{"problems": "x"}')
        with self.assertRaises(sim.SimError):
            sim.load(self.path)


class RecordTest(unittest.TestCase):
    def test_new_root_cause_opens_a_problem(self):
        memory = sim.empty()
        pid = sim.record_attempt(memory, {"new": "premature completion"}, attempt(round_no=3))
        self.assertEqual(pid, "P1")
        self.assertEqual(memory["next_id"], 2)
        problem = memory["problems"][0]
        self.assertEqual(
            (problem["id"], problem["name"], problem["opened_round"], problem["opened_run"]),
            ("P1", "premature completion", 3, "r1"),
        )
        self.assertEqual(len(problem["attempts"]), 1)

    def test_same_text_appends_instead_of_opening(self):
        memory = sim.empty()
        sim.record_attempt(memory, {"new": "premature completion"}, attempt(round_no=0))
        pid = sim.record_attempt(memory, {"new": "  premature   completion\n"}, attempt(round_no=1))
        self.assertEqual(pid, "P1")
        self.assertEqual(len(memory["problems"]), 1)
        self.assertEqual([a["round"] for a in memory["problems"][0]["attempts"]], [0, 1])

    def test_different_text_opens_another(self):
        memory = sim.empty()
        sim.record_attempt(memory, {"new": "one"}, attempt(round_no=0))
        self.assertEqual(sim.record_attempt(memory, {"new": "two"}, attempt(round_no=1)), "P2")

    def test_existing_id_appends(self):
        memory = sim.empty()
        sim.record_attempt(memory, {"new": "one"}, attempt(round_no=0))
        self.assertEqual(sim.record_attempt(memory, {"id": "P1"}, attempt(round_no=1)), "P1")
        self.assertEqual(len(memory["problems"][0]["attempts"]), 2)

    def test_unknown_id_is_an_error(self):
        with self.assertRaises(sim.SimError):
            sim.record_attempt(sim.empty(), {"id": "P9"}, attempt())

    def test_find_attempt(self):
        memory = sim.empty()
        sim.record_attempt(memory, {"new": "one"}, attempt(run="r1", round_no=0))
        sim.record_attempt(memory, {"new": "two"}, attempt(run="r1", round_no=1))
        self.assertEqual(sim.find_attempt(memory, "r1", 1), "P2")
        self.assertIsNone(sim.find_attempt(memory, "r2", 1))

    def test_latest_attempt_spans_runs(self):
        memory = sim.empty()
        self.assertIsNone(sim.latest_attempt(memory))
        sim.record_attempt(memory, {"new": "one"}, attempt(run="20261120-0900", round_no=4))
        sim.record_attempt(memory, {"new": "two"}, attempt(run="20261204-0900", round_no=0, sample="s3"))
        self.assertEqual(sim.latest_attempt(memory)["sample"], "s3")

    def test_add_hop(self):
        memory = sim.empty()
        hop = {"run": "a", "benchmark": "b-v1", "model": "m", "into": "b"}
        sim.add_hop(memory, hop)
        self.assertEqual(memory["carried_from"], [hop])


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.memory = sim.empty()
        sim.record_attempt(self.memory, {"new": "one"}, attempt(run="r1", round_no=0))
        sim.record_attempt(self.memory, {"new": "two"}, attempt(run="r1", round_no=1))
        sim.record_attempt(self.memory, {"new": "three"}, attempt(run="r1", round_no=2))
        sim.record_attempt(self.memory, {"id": "P1"}, attempt(run="r2", round_no=0))

    def test_attempts_move_in_time_order_and_an_alias_is_kept(self):
        sim.merge(self.memory, "P2", "P1")
        self.assertEqual([p["id"] for p in self.memory["problems"]], ["P1", "P3"])
        merged = sim.problem(self.memory, "P1")
        self.assertEqual([(a["run"], a["round"]) for a in merged["attempts"]], [("r1", 0), ("r1", 1), ("r2", 0)])
        self.assertEqual(self.memory["aliases"], {"P2": "P1"})
        self.assertEqual(sim.resolve(self.memory, "P2"), "P1")

    def test_old_aliases_follow_a_later_merge(self):
        sim.merge(self.memory, "P2", "P1")
        sim.merge(self.memory, "P1", "P3")
        self.assertEqual(self.memory["aliases"], {"P2": "P3", "P1": "P3"})
        self.assertEqual(len(sim.problem(self.memory, "P3")["attempts"]), 4)

    def test_cited_alias_records_on_the_surviving_problem(self):
        sim.merge(self.memory, "P2", "P1")
        self.assertEqual(sim.record_attempt(self.memory, {"id": "P2"}, attempt(run="r3")), "P1")

    def test_invalid_merges(self):
        for src, dst in (("P1", "P1"), ("P9", "P1"), ("P1", "P9")):
            with self.assertRaises(sim.SimError):
                sim.merge(self.memory, src, dst)


class OutcomeTest(unittest.TestCase):
    def test_fixed_missed_regressed(self):
        fixed, missed, regressed = sim.outcome(
            predicted=["t3", "t4"], before={"t1", "t2"}, after={"t1", "t3"},
        )
        self.assertEqual((fixed, missed, regressed), (["t3"], ["t4"], ["t2"]))

    def test_nothing_predicted(self):
        self.assertEqual(sim.outcome([], {"a"}, {"a", "b"}), ([], [], []))


class RenderTest(unittest.TestCase):
    def test_readable_text(self):
        memory = sim.empty()
        self.assertEqual(sim.render_text(memory), "The memory is empty.")
        sim.record_attempt(
            memory, {"new": "premature completion"},
            attempt(what="Add a completion gate", predicted=["t3", "t4"], fixed=["t3"], missed=["t4"]),
        )
        text = sim.render_text(memory)
        self.assertIn("P1  premature completion", text)
        self.assertIn("r1 round 0 (b-v1, m): Add a completion gate", text)
        self.assertIn("predicted t3, t4; fixed t3; missed t4; regressed none", text)
        self.assertIn("0.00 -> 50.00", text)
