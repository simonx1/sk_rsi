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


# A gh that keeps the pull requests it was asked to create, so that `pr list` can answer for them.
FAKE_GH_STATEFUL = '''#!/usr/bin/env python3
import json, os, subprocess, sys
args = sys.argv[1:]
store = os.environ["GH_PRS"]
tip = lambda branch: subprocess.run(["git", "rev-parse", branch], capture_output=True, text=True).stdout.strip()
prs = json.load(open(store)) if os.path.exists(store) else []
with open(os.environ["GH_CALLS"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")
value = lambda flag: args[args.index(flag) + 1]
if args[:2] == ["pr", "list"]:
    fields = value("--json").split(",")
    found = [p for p in prs if p["head"] == value("--head")][:int(value("--limit"))]
    print(json.dumps([{f: p.get(f) for f in fields} for p in found]))
elif args[:2] == ["pr", "create"]:
    if os.environ.get("GH_FAIL_ON") == value("--head"):
        sys.stderr.write("not logged in\\n")
        sys.exit(1)
    pr = {"head": value("--head"), "base": value("--base"), "title": value("--title"),
          "body": open(value("--body-file")).read(), "isDraft": "--draft" in args,
          "headRefOid": tip(value("--head")), "isCrossRepository": False, "state": "OPEN",
          "url": "https://example.test/pull/%d" % (len(prs) + 1)}
    json.dump(prs + [pr], open(store, "w"))
    print(pr["url"])
'''

RUN = "20261120-0900"


class PerEditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.repo = make_repo(self.tmp / "repo", {"a.txt": "one\ntwo\n", "b.txt": "x\n"})
        self.start = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "checkout", "-q", "-b", f"rsi/{RUN}")
        self.edits = []
        for name, text, subject in (("a.txt", "ONE\ntwo\n", "E0"), ("b.txt", "y\n", "E1"), ("a.txt", "uno\ntwo\n", "E2")):
            (self.repo / name).write_text(text)
            git(self.repo, "commit", "-q", "-am", subject)
            self.edits.append(git(self.repo, "rev-parse", "HEAD"))
        git(self.repo, "checkout", "-q", "main")
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH_STATEFUL)
        gh.chmod(0o755)
        self.prs, self.calls = self.tmp / "prs.json", self.tmp / "calls.jsonl"
        self.enterContext(mock.patch.dict(os.environ, {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GH_PRS": str(self.prs), "GH_CALLS": str(self.calls)}))
        self.cfg = replace(minimal_config(), repo=str(self.repo), promotion_mode="per_edit", paired=True)
        self.memory = sim.empty()
        sim.record_attempt(self.memory, {"new": "too few workers"}, {"run": RUN, "round": 0})
        self.st = self.make_state()
        self.scratch = self.tmp / "scratch"

    def entry(self, round_no, sha, title):
        return {"round": round_no, "accepted": "s1" if sha else None, "problem": "P1" if sha else None, "opened": False,
                "what": f"{title}.\nA longer explanation." if sha else None, "title": title if sha else None,
                "risk": "Workers may queue." if sha else None, "sha": sha,
                "score_before": 50.0, "score_after": 61.0 if sha else 50.0,
                "fixed": [], "missed": [], "regressed": [], "reason": "scripted", "dropped": [],
                "expected": {"wall_s": -20.0} if sha else {}, "observed": {"wall_s": -25.00, "cost_usd": 6.00} if sha else {},
                "metrics_before": {"wall_s": 800.0, "cost_usd": 4.00} if sha else {},
                "metrics_after": {"wall_s": 600.0, "cost_usd": 4.24} if sha else {}, "limits": [], "runs": 4}

    def make_state(self, heldout=None):
        st = RunState(run=RUN, project=str(self.repo), branch=f"rsi/{RUN}", base_ref="main", config={},
                      benchmark="bench-v1", rounds=4, evolve_ids=[], heldout_ids=[],
                      start_sha=self.start, current_sha=self.edits[2], eval_sha=self.edits[2], external=True)
        st.scores = [50.0, 61.0, 63.0, 63.0, 66.0]
        st.heldout = heldout or {"start": 50.1, "end": 60.9, "limits": []}
        st.rounds_log = [self.entry(0, self.edits[0], "Edit zero"), self.entry(1, self.edits[1], "Edit one"),
                         self.entry(2, None, None), self.entry(3, self.edits[2], "Edit two")]
        return st

    def add_remote(self):
        remote = self.tmp / "remote.git"
        git(self.tmp, "init", "-q", "--bare", str(remote))
        git(self.repo, "remote", "add", "origin", str(remote))
        return remote

    def go(self, st=None, cfg=None):
        return promote.promote(self.repo, st or self.st, cfg or self.cfg, self.memory, scratch=self.scratch)

    def creates(self):
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        return [c for c in calls if c[:2] == ["pr", "create"]]

    def test_branches_independent_and_stacked(self):
        got = promote.build_edit_branches(self.repo, self.st, self.cfg, self.scratch)
        self.assertEqual([(b["round"], b["branch"], b["stacked"]) for b in got],
                         [(0, f"rsi/{RUN}-r0", False), (1, f"rsi/{RUN}-r1", False), (3, f"rsi/{RUN}-r3", True)])
        self.assertEqual(git(self.repo, "diff", "--name-only", self.start, f"rsi/{RUN}-r1"), "b.txt")
        self.assertEqual(git(self.repo, "show", f"rsi/{RUN}-r1:a.txt"), "one\ntwo")  # without the earlier edit
        self.assertEqual(git(self.repo, "rev-parse", f"rsi/{RUN}-r3"), self.edits[2])
        self.assertFalse(self.scratch.exists())
        tips = [git(self.repo, "rev-parse", b["branch"]) for b in got]
        again = promote.build_edit_branches(self.repo, self.st, self.cfg, self.scratch)
        self.assertEqual(again, got)
        self.assertEqual([git(self.repo, "rev-parse", b["branch"]) for b in again], tips)

    def test_branch_prefix(self):
        cfg = replace(self.cfg, branch_prefix="sk-bench-")
        self.assertEqual(promote.edit_branch(cfg, self.st, self.st.rounds_log[1]), f"sk-bench-{RUN}-r1")

    def test_opens_one_pull_request_per_edit(self):
        remote = self.add_remote()
        result = self.go()
        self.assertEqual([(r["round"], r["url"], r["draft"], r["stacked"]) for r in result["prs"]],
                         [(0, "https://example.test/pull/1", False, False), (1, "https://example.test/pull/2", False, False),
                          (3, "https://example.test/pull/3", False, True)])
        self.assertNotIn("manual", result)
        created = json.loads(self.prs.read_text())
        self.assertEqual([(p["head"], p["base"], p["title"]) for p in created],
                         [(f"rsi/{RUN}-r0", "main", "Edit zero"), (f"rsi/{RUN}-r1", "main", "Edit one"),
                          (f"rsi/{RUN}-r3", "main", "Edit two")])
        self.assertIn("## Measured", created[0]["body"])
        self.assertIn("| wall_s | 800.00 | 600.00 | -25.00% |", created[0]["body"])
        self.assertEqual(git(remote, "rev-parse", f"refs/heads/rsi/{RUN}-r3"), self.edits[2])

    def test_draft_when_configured_or_warned(self):
        self.add_remote()
        result = self.go(cfg=replace(self.cfg, draft=True))
        self.assertTrue(all(r["draft"] for r in result["prs"]))
        self.assertTrue(all("--draft" in call for call in self.creates()))
        self.assertFalse(json.loads(self.prs.read_text())[0]["body"].startswith("> **Warning"))

    def test_a_heldout_drop_makes_drafts_with_a_warning(self):
        self.add_remote()
        st = self.make_state(heldout={"start": 33.33, "end": 16.67, "limits": []})
        result = self.go(st=st)
        self.assertTrue(all(r["draft"] for r in result["prs"]))
        body = json.loads(self.prs.read_text())[0]["body"]
        self.assertTrue(body.startswith("> **Warning:** the held-out score went down (33.33 → 16.67)"), body[:120])

    def test_a_heldout_limit_makes_drafts_with_a_warning(self):
        over = {"metric": "cost_usd", "direction": "increase", "allowed_pct": 10.0, "total_pct": 12.3, "state": "over"}
        st = self.make_state(heldout={"start": 50.0, "end": 60.0, "limits": [over]})
        self.assertEqual(promote.run_warning(st), "the whole run is past a limit: cost_usd +12.30% of +10.00% allowed")
        self.assertIsNone(promote.run_warning(self.st))

    def test_second_promote_creates_only_what_is_missing(self):
        self.add_remote()
        with mock.patch.dict(os.environ, {"GH_FAIL_ON": f"rsi/{RUN}-r1"}):
            first = self.go()
        self.assertEqual([sorted(k for k in r if k in ("url", "error")) for r in first["prs"]], [["url"], ["error"], ["url"]])
        self.assertEqual(first["prs"][1]["error"], "gh pr create failed: not logged in")
        self.st.promoted = first
        second = self.go()
        self.assertEqual([r.get("url") for r in second["prs"]],
                         ["https://example.test/pull/1", "https://example.test/pull/3", "https://example.test/pull/2"])
        self.assertEqual([c[c.index("--head") + 1] for c in self.creates()],
                         [f"rsi/{RUN}-r0", f"rsi/{RUN}-r1", f"rsi/{RUN}-r3", f"rsi/{RUN}-r1"])

    def test_an_existing_pull_request_is_reused(self):
        self.add_remote()
        promote.build_edit_branches(self.repo, self.st, self.cfg, self.scratch)
        tip = git(self.repo, "rev-parse", f"rsi/{RUN}-r0")
        self.prs.write_text(json.dumps([{"head": f"rsi/{RUN}-r0", "url": "https://example.test/pull/77", "isDraft": True,
                                         "headRefOid": tip, "isCrossRepository": False, "state": "CLOSED"}]))
        result = self.go()
        self.assertEqual((result["prs"][0]["url"], result["prs"][0]["draft"]), ("https://example.test/pull/77", True))
        self.assertEqual(len(self.creates()), 2)

    def test_a_pull_request_that_is_not_ours_is_not_reused(self):
        """Same branch name, but from a fork, or for another commit: it is somebody else's pull request."""
        self.add_remote()
        promote.build_edit_branches(self.repo, self.st, self.cfg, self.scratch)
        tip = git(self.repo, "rev-parse", f"rsi/{RUN}-r0")
        others = [
            {"head": f"rsi/{RUN}-r0", "url": "https://example.test/pull/66", "isDraft": False,
             "headRefOid": tip, "isCrossRepository": True, "state": "OPEN"},
            {"head": f"rsi/{RUN}-r0", "url": "https://example.test/pull/67", "isDraft": False,
             "headRefOid": "0" * 40, "isCrossRepository": False, "state": "OPEN"},
        ]
        self.prs.write_text(json.dumps(others))
        result = self.go()
        self.assertNotIn(result["prs"][0]["url"], ("https://example.test/pull/66", "https://example.test/pull/67"))
        self.assertEqual(len(self.creates()), 3)

    def test_without_a_remote_branches_exist_and_commands_are_printed(self):
        result = self.go()
        self.assertEqual([r["skipped"] for r in result["prs"]], ["the project has no git remote"] * 3)
        self.assertEqual(git(self.repo, "rev-parse", f"rsi/{RUN}-r3"), self.edits[2])
        self.assertEqual(result["manual"][0], f"git push -u origin rsi/{RUN}-r0")
        self.assertEqual(result["manual"][1], f"gh pr create --base main --head rsi/{RUN}-r0 --title 'Edit zero'")
        self.assertEqual(len(result["manual"]), 6)
        self.assertFalse(self.calls.exists())

    def test_nothing_accepted(self):
        self.st.rounds_log = [self.entry(0, None, None)]
        self.assertEqual(self.go(), {"skipped": "no edit was accepted in this run"})

    def test_title_falls_back_to_the_first_line(self):
        entry = dict(self.st.rounds_log[0], title=None, what="x" * 80 + "\nmore")
        self.assertEqual(promote.edit_title(entry), "x" * 69 + "...")
        self.assertEqual(promote.edit_title(self.st.rounds_log[0]), "Edit zero")

    def test_body_sections(self):
        log = self.st.rounds_log
        body = promote.edit_body(self.st, self.cfg, self.memory, log[3], earlier=[log[0], log[1]], stacked=True)
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        self.assertEqual(headings, ["## What", "## Measured", "## Predicted and measured", "## Risk",
                                    "## Measured on top of", "## Whole run", "## Memory"])
        for line in ("Edit two.", "A longer explanation.", "| | Before | After | Change |",
                     "| Score | 50.00 | 61.00 | +11.00 |", "| wall_s | 800.00 | 600.00 | -25.00% |",
                     "| cost_usd | 4.00 | 4.24 | +6.00% |",
                     "Each side: 4 runs per task, measured in the same batch in round 3.",
                     "- wall_s: predicted -20.00%, measured -25.00%", "Workers may queue.",
                     "- Edit zero", "- Edit one", "This branch contains those edits. Merge them first.",
                     "Held-out: 50.10 → 60.90", "**P1**: too few workers"):
            self.assertIn(line, body)
        first = promote.edit_body(self.st, self.cfg, self.memory, log[0], earlier=[], stacked=False)
        self.assertIn("Nothing: this edit was measured on the starting version.", first)
        self.assertNotIn("Merge them first", first)
        unpaired = promote.edit_body(self.st, replace(self.cfg, paired=False), self.memory, log[0], earlier=[], stacked=False)
        self.assertIn("Each side: 4 runs per task. The baseline was measured before round 0.", unpaired)
