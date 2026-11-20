import json
import os
import subprocess
import sys
from pathlib import Path

from rsi import config, evaluator, tasks
from rsi.state import Layout
from tests.toy import TOY, ToyCase


class ToyProjectTest(ToyCase):
    def test_starting_scores(self):
        project = self.make_toy()
        cfg = config.load(project / ".rsi" / "config.toml")
        layout = Layout(cfg.name)
        all_tasks = tasks.load_tasks(layout.tasks_path(cfg.tasks_file))
        split = tasks.load_split(layout, [t["id"] for t in all_tasks], cfg.heldout_fraction)
        evolve = [t for t in all_tasks if split[t["id"]] == "evolve"]
        heldout = [t for t in all_tasks if split[t["id"]] == "heldout"]
        self.assertEqual([t["id"] for t in evolve], ["t01", "t02", "t03", "t04", "t05", "t06"])
        res = evaluator.evaluate(cfg, evolve, project, self.tmp / "results" / "evolve", "toy-v1")
        self.assertEqual((res.score(), res.passing(0.5)), (33.33, {"t01", "t02"}))
        self.assertEqual(res.infra_failure_rate(), 0.0)
        res = evaluator.evaluate(cfg, heldout, project, self.tmp / "results" / "heldout", "toy-v1")
        self.assertEqual((res.score(), res.passing(0.5)), (33.33, {"t07", "t08"}))

    def test_scoring_leaves_the_checkout_clean(self):
        project = self.make_toy()
        cfg = config.load(project / ".rsi" / "config.toml")
        evaluator.evaluate(cfg, [{"id": "t01", "text": "refund my invoice", "label": "billing"}],
                           project, self.tmp / "results", "toy-v1")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=project, capture_output=True, text=True)
        self.assertEqual(status.stdout, "")


class FakeAgentTest(ToyCase):
    def call(self, script, env):
        work = self.tmp / "work"
        work.mkdir(exist_ok=True)
        (work / "rules.txt").write_text("invoice => billing\ncrash => bug\n")
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        full = dict(os.environ, FAKE_AGENT_SCRIPT=str(TOY / script), **env)
        done = subprocess.run([sys.executable, str(TOY / "fake_agent.py"), "--out", str(out)], cwd=work, env=full)
        return done.returncode, work, out

    def test_proposer_edits_and_writes_a_manifest(self):
        code, work, out = self.call("script.json", {"RSI_ROLE": "proposer", "RSI_ROUND": "0", "RSI_SAMPLE": "s1"})
        self.assertEqual(code, 0)
        self.assertTrue((work / "rules.txt").read_text().endswith("charged => billing\nerror => bug\n"))
        self.assertEqual(json.loads((out / "manifest.json").read_text())["predicted"], ["t03", "t05"])

    def test_raw_manifest_is_written_as_is(self):
        _, _, out = self.call("script.json", {"RSI_ROLE": "proposer", "RSI_ROUND": "0", "RSI_SAMPLE": "s3"})
        self.assertEqual((out / "manifest.json").read_text(), "I added a rule for 'twice'.")

    def test_missing_entry_exits_3(self):
        code, _, _ = self.call("script.json", {"RSI_ROLE": "proposer", "RSI_ROUND": "9", "RSI_SAMPLE": "s1"})
        self.assertEqual(code, 3)
        code, _, _ = self.call("script_selector_down.json",
                               {"RSI_ROLE": "selector", "RSI_ROUND": "0", "RSI_STAGE": "screen"})
        self.assertEqual(code, 3)

    def test_selector_answers(self):
        _, _, out = self.call("script.json", {"RSI_ROLE": "selector", "RSI_ROUND": "1", "RSI_STAGE": "screen"})
        self.assertEqual(json.loads((out / "screen.json").read_text())["screen"], [
            {"sample": "s1", "keep": True, "reason": "scripted"},
            {"sample": "s2", "keep": True, "reason": "scripted"},
        ])
        _, _, out = self.call("script.json", {"RSI_ROLE": "selector", "RSI_ROUND": "1", "RSI_STAGE": "select"})
        self.assertEqual(json.loads((out / "decision.json").read_text()), {"accept": "s1", "reason": "scripted"})
