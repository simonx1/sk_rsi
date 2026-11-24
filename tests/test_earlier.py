import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rsi import earlier
from rsi.state import Layout


class NotKeptTest(unittest.TestCase):
    def setUp(self):
        home = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(home)}))
        self.layout = Layout("proj")

    def candidate(self, run, round_no, sample, manifest=None, raw_result=None, **result):
        folder = self.layout.run(run).history / f"round-{round_no}" / sample
        folder.mkdir(parents=True, exist_ok=True)
        if manifest is not False:
            manifest = manifest or {"what": f"Edit {sample} of round {round_no}", "title": f"{sample} title"}
            (folder / "manifest.json").write_text(json.dumps(manifest))
        if raw_result is not None:
            (folder / "result.json").write_text(raw_result)
        else:
            (folder / "result.json").write_text(json.dumps({"sample": sample, **result}))

    def decide(self, run, round_no, accept, reason="why"):
        folder = self.layout.run(run).history / f"round-{round_no}"
        (folder / "decision.json").write_text(json.dumps({"accept": accept, "reason": reason}))

    def log(self, run, *entries):
        path = self.layout.run(run).state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rounds_log": list(entries)}))

    def outcomes(self, run="R9", round_no=0, **kwargs):
        entries, omitted = earlier.not_kept(self.layout, run, round_no, **kwargs)
        return [(e["folder"], e["title"], e["outcome"]) for e in entries], omitted

    def test_dropped_evaluated_guarded_undecided_and_kept(self):
        self.candidate("R1", 0, "s1", status="evaluated", score=0.0, observed={},
                       table=[{"task": "ci", "passes": False, "summary": "not run: protected task changed"}])
        self.candidate("R1", 0, "s2", status="dropped", reason="screened out: duplicates s1")
        self.candidate("R1", 0, "s3", status="evaluated", score=91.0, observed={"wall_s": -2.5, "cost_usd": 1.0},
                       table=[{"task": "ci", "passes": True, "summary": "wall 610 s"}])
        self.decide("R1", 0, None, "guard: s3 was measured again: it gained 1.00 points, less than the 2.50 "
                                   "required; the selector's reason was: s3 is fastest")
        self.log("R1", {"round": 0, "score_before": 87.27})
        self.candidate("R1", 1, "s1", status="evaluated", score=95.0, observed={"wall_s": -9.0})
        self.decide("R1", 1, "s1")
        self.log("R1", {"round": 0, "score_before": 87.27}, {"round": 1, "score_before": 89.0})
        self.candidate("R1", 2, "s1", status="kept")
        self.candidate("R1", 2, "s2", status="evaluated", score=80.0, observed={})
        found, omitted = self.outcomes()
        self.assertEqual(omitted, 0)
        self.assertEqual(found, [
            ("R1/history/round-2/s1", "s1 title", "the run stopped before this round was decided"),
            ("R1/history/round-2/s2", "s2 title", "scored 80.00; the run stopped before this round was decided"),
            ("R1/history/round-0/s1", "s1 title",
             "scored 0.00 against 87.27 for the version it changed; ci: not run: protected task changed"),
            ("R1/history/round-0/s2", "s2 title", "dropped: screened out: duplicates s1"),
            ("R1/history/round-0/s3", "s3 title",
             "scored 91.00 against 87.27 for the version it changed; measured wall_s -2.50%, cost_usd +1.00%; "
             "guard: s3 was measured again: it gained 1.00 points, less than the 2.50 required; ci: wall 610 s"),
        ])

    def test_a_round_that_compared_nothing_says_so(self):
        self.candidate("R1", 0, "s1", status="evaluated", score=75.0, observed={})
        self.decide("R1", 0, None, "the round's baseline had failed runs; nothing can be compared")
        self.candidate("R1", 1, "s1", status="evaluated", score=70.0, observed={})
        self.decide("R1", 1, None, "selector failed: the agent timed out")
        self.candidate("R1", 2, "s1", status="evaluated", score=60.0, observed={})
        self.decide("R1", 2, None, "s1 is no better; long free text from the selector")
        found = {f: o for f, _, o in self.outcomes()[0]}
        self.assertEqual(found["R1/history/round-0/s1"],
                         "scored 75.00; not compared: the round's baseline had failed runs; nothing can be compared")
        self.assertEqual(found["R1/history/round-1/s1"], "scored 70.00; not compared: selector failed: the agent timed out")
        self.assertEqual(found["R1/history/round-2/s1"], "scored 60.00")

    def test_damaged_history_never_stops_a_round(self):
        self.candidate("R1", 0, "s1", status="evaluated", score=50.0, observed=7, table=5)
        self.candidate("R1", 0, "s2", status="dropped", reason=["not", "text"])
        self.candidate("R1", 0, "s3", raw_result="[" * 100000 + "]" * 100000)
        self.decide("R1", 0, None)
        path = self.layout.run("R1").state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"rounds_log": null}')
        self.candidate("R2", 0, "s1", status="evaluated", score=40.0)
        (self.layout.run("R2").history / "round-0" / "decision.json").write_text("[]")
        self.candidate("R3", 0, "s1", status="dropped", reason="fine")
        locked = self.layout.run("R3").history / "round-0"
        locked.chmod(0)
        self.addCleanup(locked.chmod, 0o755)
        found = {f: o for f, _, o in self.outcomes("R9")[0]}
        self.assertEqual(found["R1/history/round-0/s1"], "scored 50.00")
        self.assertEqual(found["R1/history/round-0/s2"], "dropped: ['not', 'text']")
        self.assertEqual(found["R2/history/round-0/s1"], "scored 40.00; the run stopped before this round was decided")
        self.assertNotIn("R1/history/round-0/s3", found)
        self.assertNotIn("R3/history/round-0/s1", found)

    def test_failing_task_notes_come_first_and_at_most_three(self):
        rows = [{"task": f"t{n}", "passes": n % 2 == 0, "summary": f"note {n}"} for n in range(1, 7)]
        rows.append({"task": "t9", "passes": False, "summary": ""})
        self.candidate("R1", 0, "s1", status="evaluated", score=50.0, observed={}, table=rows)
        self.decide("R1", 0, None)
        (_, _, outcome), = self.outcomes()[0]
        self.assertEqual(outcome, "scored 50.00; t1: note 1; t3: note 3; t5: note 5")

    def test_skipped_entries(self):
        self.candidate("R1", 0, "s1", manifest=False, status="dropped", reason="the proposer failed")
        self.candidate("R1", 0, "s2", raw_result="{not json", status="dropped")
        self.candidate("R1", 0, "s3", status="dropped", reason="kept out")
        (self.layout.run("R1").history / "round-0" / "s4").mkdir()      # no files at all
        (self.layout.run("R1").history / "round-0" / "notes.txt").write_text("not a candidate")
        self.decide("R1", 0, None)
        self.assertEqual([f for f, _, _ in self.outcomes()[0]], ["R1/history/round-0/s3"])

    def test_the_round_under_way_is_left_out(self):
        for round_no in (0, 1, 2):
            self.candidate("R2", round_no, "s1", status="dropped", reason=f"r{round_no}")
        self.assertEqual([f for f, _, _ in self.outcomes("R2", 1)[0]], ["R2/history/round-0/s1"])
        self.assertEqual(len(self.outcomes("R3", 0)[0]), 3)     # another run: all of R2's rounds

    def test_newest_first_across_runs_and_rounds(self):
        for run in ("20261120-0900", "20261201-0930"):
            for round_no in (2, 10):
                for sample in ("s2", "s1"):
                    self.candidate(run, round_no, sample, status="dropped", reason="x")
        folders = [f for f, _, _ in self.outcomes("20261125-0800")[0]]
        self.assertEqual(folders, [
            "20261201-0930/history/round-10/s1", "20261201-0930/history/round-10/s2",
            "20261201-0930/history/round-2/s1", "20261201-0930/history/round-2/s2",
            "20261120-0900/history/round-10/s1", "20261120-0900/history/round-10/s2",
            "20261120-0900/history/round-2/s1", "20261120-0900/history/round-2/s2",
        ])

    def test_limit_and_cut(self):
        for round_no in range(12):
            for sample in ("s1", "s2", "s3"):
                self.candidate("R1", round_no, sample, status="dropped", reason="screened out: " + "x" * 400)
        found, omitted = self.outcomes(limit=30)
        self.assertEqual((len(found), omitted), (30, 6))
        outcome = found[0][2]
        self.assertEqual(len(outcome), 300)
        self.assertTrue(outcome.endswith("…"))

    def test_title_falls_back_to_the_first_line_of_what(self):
        self.candidate("R1", 0, "s1", manifest={"what": "Cache the parsed rules " + "y" * 80 + "\nand more"},
                       status="dropped", reason="x")
        title = self.outcomes()[0][0][1]
        self.assertEqual(len(title), 72)
        self.assertTrue(title.startswith("Cache the parsed rules y"))
        self.assertTrue(title.endswith("…"))

    def test_no_runs_yet(self):
        self.assertEqual(self.outcomes(), ([], 0))


if __name__ == "__main__":
    unittest.main()
