import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rsi import promote, sim
from rsi.state import RunState
from tests.helpers import git, make_repo, minimal_config

FAKE_GH = '''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
body = open(args[args.index("--body-file") + 1]).read()
with open(os.environ["GH_LOG"], "w") as fh:
    json.dump({"args": args, "body": body}, fh)
if os.environ.get("GH_FAIL"):
    sys.stderr.write("not logged in\\n")
    sys.exit(1)
print("Creating pull request")
print("https://example.test/toy/pull/1")
'''

BRANCH = "rsi/20261120-0900"


def make_state(heldout_end=66.67, accepted=True):
    st = RunState(
        run="20261120-0900", project="/p", branch=BRANCH, base_ref="main", config={},
        benchmark="toy-v1", rounds=2, evolve_ids=[], heldout_ids=[],
        start_sha="a", current_sha="b", eval_sha="b",
    )
    st.scores = [33.33, 66.67, 66.67]
    st.heldout = {"start": 33.33, "end": heldout_end}
    st.rounds_log = [
        {"round": 0, "accepted": "s1" if accepted else None, "problem": "P1" if accepted else None,
         "opened": accepted, "what": "Add the keywords | charged and error" if accepted else None,
         "score_before": 33.33, "score_after": 66.67, "fixed": ["t03", "t05"], "missed": [],
         "regressed": [], "reason": "scripted", "dropped": []},
        {"round": 1, "accepted": None, "problem": None, "opened": False, "what": None,
         "score_before": 66.67, "score_after": 66.67, "fixed": [], "missed": [], "regressed": [],
         "reason": "guard: s1 scored 0.00", "dropped": []},
    ]
    return st


def make_memory():
    memory = sim.empty()
    sim.record_attempt(memory, {"new": "rules cover too few keywords"}, {"run": "20261120-0900", "round": 0})
    return memory


class TextTest(unittest.TestCase):
    def test_title(self):
        self.assertEqual(
            promote.pr_title(make_state()),
            "RSI 20261120-0900: evolve 33.33 → 66.67, held-out 33.33 → 66.67",
        )

    def test_body(self):
        body = promote.pr_body(make_state(), minimal_config(), make_memory())
        self.assertFalse(body.startswith("> **Warning"))
        self.assertIn("- Benchmark: toy-v1", body)
        self.assertIn("- Model: model-a", body)
        self.assertIn("| Evolve | 33.33 | 66.67 |", body)
        self.assertIn("| Held-out | 33.33 | 66.67 |", body)
        self.assertIn("| 0 | s1 (P1): Add the keywords / charged and error | 33.33 → 66.67 | t03, t05 | none | none |", body)
        self.assertIn("| 1 | none: guard: s1 scored 0.00 | 66.67 → 66.67 | none | none | none |", body)
        self.assertIn("- **P1** (opened): rules cover too few keywords. 1 attempt in this run, 1 in total.", body)

    def test_body_warns_when_heldout_dropped(self):
        body = promote.pr_body(make_state(heldout_end=16.67), minimal_config(), make_memory())
        self.assertTrue(body.startswith("> **Warning:** the held-out score went down (33.33 → 16.67)"))


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.repo = make_repo(self.tmp / "repo", {"rules.txt": "a => b\n"})
        git(self.repo, "branch", BRANCH)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        self.gh_log = self.tmp / "gh.json"
        self.enterContext(mock.patch.dict(os.environ, {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GH_LOG": str(self.gh_log)}))
        self.cfg = minimal_config()

    def add_remote(self):
        remote = self.tmp / "remote.git"
        git(self.tmp, "init", "-q", "--bare", str(remote))
        git(self.repo, "remote", "add", "origin", str(remote))
        return remote

    def go(self, st=None, cfg=None):
        return promote.promote(self.repo, st or make_state(), cfg or self.cfg, make_memory())

    def test_nothing_accepted_means_no_pull_request(self):
        self.add_remote()
        self.assertEqual(self.go(make_state(accepted=False)), {"skipped": "no edit was accepted in this run"})
        self.assertFalse(self.gh_log.exists())

    def test_switched_off_in_config(self):
        self.add_remote()
        result = self.go(cfg=replace(self.cfg, pull_request=False))
        self.assertEqual(result["skipped"], "promotion.pull_request is false")
        self.assertEqual(result["manual"][0], f"git push -u origin {BRANCH}")
        self.assertTrue(result["manual"][1].startswith(f"gh pr create --base main --head {BRANCH} --title "))

    def test_no_remote(self):
        self.assertEqual(self.go()["skipped"], "the project has no git remote")

    def test_no_gh(self):
        self.add_remote()
        with mock.patch("rsi.promote.shutil.which", return_value=None):
            self.assertEqual(self.go()["skipped"], "gh is not installed")

    def test_pushes_and_opens_a_pull_request(self):
        remote = self.add_remote()
        self.assertEqual(self.go(), {"url": "https://example.test/toy/pull/1", "draft": False})
        self.assertEqual(git(remote, "rev-parse", f"refs/heads/{BRANCH}"), git(self.repo, "rev-parse", BRANCH))
        call = json.loads(self.gh_log.read_text())
        self.assertEqual(call["args"][:8], [
            "pr", "create", "--base", "main", "--head", BRANCH,
            "--title", "RSI 20261120-0900: evolve 33.33 → 66.67, held-out 33.33 → 66.67"])
        self.assertNotIn("--draft", call["args"])
        self.assertIn("| Held-out | 33.33 | 66.67 |", call["body"])

    def test_heldout_drop_opens_a_draft(self):
        self.add_remote()
        result = self.go(make_state(heldout_end=16.67))
        self.assertTrue(result["draft"])
        call = json.loads(self.gh_log.read_text())
        self.assertIn("--draft", call["args"])
        self.assertTrue(call["body"].startswith("> **Warning:**"))

    def test_push_failure_is_reported(self):
        git(self.repo, "remote", "add", "origin", str(self.tmp / "missing.git"))
        result = self.go()
        self.assertTrue(result["error"].startswith("push failed"))
        self.assertIn("manual", result)

    def test_gh_failure_is_reported(self):
        self.add_remote()
        with mock.patch.dict(os.environ, {"GH_FAIL": "1"}):
            result = self.go()
        self.assertEqual(result["error"], "gh pr create failed: not logged in")
