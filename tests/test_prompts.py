import json
import unittest
from pathlib import Path

from rsi import prompts, sim
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
