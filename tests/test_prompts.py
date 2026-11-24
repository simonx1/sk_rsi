import json
import unittest
from pathlib import Path

from rsi import prompts, sim
from rsi.config import Constraint
from tests.helpers import minimal_config

MEMORY = sim.empty()
sim.record_attempt(MEMORY, {"new": "premature completion"}, {
    "run": "r1", "round": 0, "sample": "s1", "accepted": True, "backfilled": False,
    "what": "Add a completion gate", "files": ["rules.txt"], "predicted": ["t3"],
    "fixed": [], "missed": ["t3"], "regressed": [], "passed": 2, "total": 6,
    "score_before": 33.33, "score_after": 33.33, "domain": {"benchmark": "toy-v1", "model": "model-a"},
})
TABLE = [
    {"task": "t01", "score": 1.0, "passes": True, "errors": 0, "summary": "correct"},
    {"task": "t03", "score": 0.0, "passes": False, "errors": 1, "summary": "got other | expected billing"},
]
CFG = minimal_config(editable=("rules.txt", "prompts/**"), frozen=("score.py",))


class ProposerPromptTest(unittest.TestCase):
    def build(self, evidence=()):
        return prompts.proposer_prompt(
            CFG, "toy-v1", MEMORY, TABLE, list(evidence), 33.33, "abc123",
            Path("/state/feedback"), Path("/state/history"), Path("/state/out"))

    def test_carries_the_whole_memory(self):
        text = self.build()
        self.assertIn(json.dumps(MEMORY, indent=2), text)

    def test_carries_all_eight_rules(self):
        text = self.build()
        self.assertEqual(len(prompts.SIM_RULES), 8)
        for number, rule in enumerate(prompts.SIM_RULES, 1):
            self.assertIn(f"{number}. {rule}", text)

    def test_states_setting_paths_and_results(self):
        text = self.build()
        self.assertIn("Benchmark: toy-v1", text)
        self.assertIn("Model that runs the tasks: model-a", text)
        self.assertIn("33.33", text)
        self.assertIn("`rules.txt`, `prompts/**`", text)
        self.assertIn("`score.py`", text)
        self.assertIn("| t01 | 1.00 | yes | 0 | correct |", text)
        self.assertIn("| t03 | 0.00 | no | 1 | got other / expected billing |", text)
        self.assertIn("/state/feedback/abc123/", text)
        self.assertIn("/state/history/", text)
        self.assertIn("/state/out/manifest.json", text)

    def test_evidence_section(self):
        self.assertIn("None since the last round.", self.build())
        text = self.build([{"id": "e1", "at": "2026-11-19T10:00:00Z", "summary": "Finding refuted", "task_id": "t03"}])
        self.assertIn("- 2026-11-19T10:00:00Z (task t03): Finding refuted", text)
        self.assertIn("not comparable between candidates", text)


class SelectorPromptTest(unittest.TestCase):
    ITEM = {
        "sample": "s2", "manifest": {"root_cause": {"id": "P1"}, "what": "Tighten the gate",
                                     "predicted": ["t03"], "risk": "none"},
        "files": ["rules.txt"], "diff": "+charged => billing",
        "score": 50.0, "fixed": ["t03"], "missed": [], "regressed": ["t01"], "table": TABLE,
    }

    def test_screen_prompt(self):
        text = prompts.screen_prompt(CFG, "toy-v1", MEMORY, [self.ITEM], Path("/state/out"))
        self.assertIn(json.dumps(MEMORY, indent=2), text)
        self.assertIn("## Candidate s2", text)
        self.assertIn("Root cause: P1", text)
        self.assertIn("+charged => billing", text)
        self.assertIn("/state/out/screen.json", text)
        self.assertIn(f"1. {prompts.SIM_RULES[0]}", text)

    def test_select_prompt(self):
        text = prompts.select_prompt(CFG, "toy-v1", MEMORY, 33.33, TABLE, [self.ITEM], Path("/state/out"))
        self.assertIn("Current harness: 33.33", text)
        self.assertIn("## Candidate s2: 50.00 (+16.67)", text)
        self.assertIn("Predicted and fixed: t03", text)
        self.assertIn("Regressed: t01", text)
        self.assertIn("/state/out/decision.json", text)
        self.assertIn("more than 1.00 points below", text)

    def test_new_root_cause_is_shown_as_text(self):
        item = dict(self.ITEM, manifest=dict(self.ITEM["manifest"], root_cause={"new": "weak retrieval"}))
        text = prompts.screen_prompt(CFG, "toy-v1", MEMORY, [item], Path("/state/out"))
        self.assertIn("Root cause: new: weak retrieval", text)


class NotesPromptTest(unittest.TestCase):
    NOTES = "The deploy task is off limits.\nRuns vary by about 4%.\n"

    def test_every_prompt_carries_the_notes(self):
        item = SelectorPromptTest.ITEM
        texts = {
            "proposer": prompts.proposer_prompt(CFG, "toy-v1", MEMORY, TABLE, [], 33.33, "abc123", Path("/f"),
                                                Path("/h"), Path("/o"), notes=self.NOTES),
            "screen": prompts.screen_prompt(CFG, "toy-v1", MEMORY, [item], Path("/o"), notes=self.NOTES),
            "select": prompts.select_prompt(CFG, "toy-v1", MEMORY, 33.33, TABLE, [item], Path("/o"), notes=self.NOTES),
        }
        for role, text in texts.items():
            with self.subTest(role=role):
                self.assertIn("# Notes from the project's owner", text)
                self.assertIn("The deploy task is off limits.\nRuns vary by about 4%.", text)
        proposer = texts["proposer"]
        self.assertLess(proposer.index("# What you may change"), proposer.index("# Notes from the project's owner"))
        self.assertLess(proposer.index("# Notes from the project's owner"), proposer.index("# Memory of earlier"))

    def test_no_notes_no_section(self):
        for text in (prompts.proposer_prompt(CFG, "toy-v1", MEMORY, TABLE, [], 33.33, "abc123", Path("/f"),
                                             Path("/h"), Path("/o")),
                     prompts.screen_prompt(CFG, "toy-v1", MEMORY, [SelectorPromptTest.ITEM], Path("/o"), notes="  \n")):
            self.assertNotIn("Notes from the project's owner", text)


class EarlierAttemptsPromptTest(unittest.TestCase):
    ENTRIES = [{"run": "R1", "folder": "R1/history/round-0/s1", "sample": "s1", "title": "Cache the rules",
                "outcome": "dropped: screened out: duplicates s2"}]

    def test_section(self):
        text = prompts.earlier_section(self.ENTRIES, 2, Path("/state/runs"))
        self.assertTrue(text.startswith("# Earlier attempts that were not kept"))
        self.assertIn('- R1/history/round-0/s1 "Cache the rules": dropped: screened out: duplicates s2', text)
        self.assertIn("`/state/runs/<run>/history/round-<n>/<sample>/`", text)
        self.assertIn("2 older ones are not listed", text)
        self.assertNotIn("older ones", prompts.earlier_section(self.ENTRIES, 0, Path("/state/runs")))
        self.assertIsNone(prompts.earlier_section([], 0, Path("/state/runs")))

    def test_proposer_and_screen_prompts_carry_it(self):
        earlier = (self.ENTRIES, 0, Path("/state/runs"))
        proposer = prompts.proposer_prompt(CFG, "toy-v1", MEMORY, TABLE, [], 33.33, "abc123", Path("/f"),
                                           Path("/h"), Path("/o"), earlier=earlier)
        self.assertIn("# Earlier attempts that were not kept", proposer)
        self.assertLess(proposer.index("# Memory of earlier"), proposer.index("# Earlier attempts that were not kept"))
        self.assertLess(proposer.index("# Earlier attempts that were not kept"), proposer.index("# Current results"))
        screen = prompts.screen_prompt(CFG, "toy-v1", MEMORY, [SelectorPromptTest.ITEM], Path("/o"), earlier=earlier)
        self.assertIn("# Earlier attempts that were not kept", screen)
        self.assertIn("it repeats an attempt that the memory, or the list of earlier attempts below, shows did "
                      "not work and brings no new evidence", screen)
        plain = prompts.screen_prompt(CFG, "toy-v1", MEMORY, [SelectorPromptTest.ITEM], Path("/o"))
        self.assertNotIn("Earlier attempts that were not kept", plain)
        self.assertNotIn("earlier attempts below", plain)
        self.assertIn("it repeats an attempt that the memory shows did not work and brings", plain)


class MeasuredGoalPromptTest(unittest.TestCase):
    ROWS = [{"task": "pr", "score": 0.5, "passes": True, "errors": 0, "summary": "13.3 min",
             "metrics": {"wall_s": 800.0, "cost_usd": 4.00}}]
    LIMITS = (Constraint("cost_usd", "increase", 10.0), Constraint("accuracy_pct", "decrease", 0.0))

    def test_proposer_prompt_shows_measurements_and_limits(self):
        text = prompts.proposer_prompt(
            CFG, "bench-v1", MEMORY, self.ROWS, [], 50.0, "abc123", Path("/s/feedback"), Path("/s/history"), Path("/s/out"),
            metrics={"wall_s": 800.0, "cost_usd": 4.00}, limits=self.LIMITS, used={"cost_usd": 1.06})
        self.assertIn("# Measurements and limits", text)
        self.assertIn("- wall_s: 800.00", text)
        self.assertIn("- cost_usd: 4.00", text)
        self.assertIn("- cost_usd may rise at most 10.00% in this run; +6.00% used", text)
        self.assertIn("- accuracy_pct may fall at most 0.00% in this run; +0.00% used", text)
        self.assertIn("| Task | Score | Passes | Run errors | wall_s | cost_usd | Note |", text)
        self.assertIn("| pr | 0.50 | yes | 0 | 800.00 | 4.00 | 13.3 min |", text)
        self.assertIn('"title": "A short name for the edit"', text)
        self.assertIn('"expected": {"wall_s": -10}', text)

    def test_a_project_without_metrics_sees_no_such_section(self):
        text = prompts.proposer_prompt(CFG, "toy-v1", MEMORY, TABLE, [], 33.33, "abc123",
                                       Path("/s/feedback"), Path("/s/history"), Path("/s/out"))
        self.assertNotIn("# Measurements and limits", text)
        self.assertNotIn('"expected"', text)

    def test_select_prompt_shows_each_candidates_measurements(self):
        item = {"sample": "s1", "manifest": {"root_cause": {"id": "P1"}, "what": "More workers", "predicted": [],
                                             "risk": "", "title": "Use 24 workers", "expected": {"wall_s": -20.0}},
                "files": ["conf/run.yml"], "score": 61.0, "fixed": [], "missed": [], "regressed": [], "table": self.ROWS,
                "metrics": {"wall_s": 600.0, "cost_usd": 4.24}, "observed": {"wall_s": -25.00, "cost_usd": 6.00},
                "limits": [{"metric": "cost_usd", "direction": "increase", "allowed_pct": 10.0, "total_pct": 6.00, "state": "ok"},
                           {"metric": "accuracy_pct", "direction": "decrease", "allowed_pct": 0.0, "total_pct": None, "state": "missing"}]}
        text = prompts.select_prompt(CFG, "bench-v1", MEMORY, 50.0, self.ROWS, [item], Path("/s/out"))
        self.assertIn("Measured: wall_s 600.00 (-25.00%), cost_usd 4.24 (+6.00%)", text)
        self.assertIn("Expected by the proposer: wall_s -20.00%", text)
        self.assertIn("Limits: cost_usd +6.00% of +10.00% allowed; accuracy_pct was not reported by the candidate", text)
        self.assertIn("Cannot be accepted: accuracy_pct was not reported by the candidate", text)
