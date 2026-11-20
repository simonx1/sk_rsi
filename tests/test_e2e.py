"""The whole loop on the toy project, with the scripted agent."""

import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

from rsi import evaluator, loop, tasks
from rsi.state import Layout
from tests.helpers import git
from tests.toy import TOY, ToyCase


def history(st, round_no, name):
    path = Layout("toy").run(st.run).history / f"round-{round_no}" / name
    return json.loads(path.read_text())


class TwoRoundsTest(ToyCase):
    def setUp(self):
        super().setUp()
        self.project = self.make_toy()
        self.st = loop.run(self.project)
        self.rl = Layout("toy").run(self.st.run)

    def test_score_improves_and_the_owner_checkout_is_untouched(self):
        st = self.st
        self.assertEqual((st.status, st.round), ("done", 2))
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33})
        self.assertEqual(st.promoted["skipped"], "the project has no git remote")
        self.assertEqual(
            git(self.project, "show", f"{st.branch}:rules.txt").splitlines(),
            ["invoice => billing", "crash => bug", "charged => billing", "error => bug", "add => feature"],
        )
        self.assertEqual(git(self.project, "status", "--porcelain"), "")
        self.assertEqual(git(self.project, "rev-parse", "--abbrev-ref", "HEAD"), "main")
        self.assertEqual((self.project / "rules.txt").read_text(), "invoice => billing\ncrash => bug\n")

    def test_memory_records_predictions_and_outcomes(self):
        st = self.st
        memory = json.loads(git(self.project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual((memory["next_id"], memory["carried_from"], memory["aliases"]), (2, [], {}))
        self.assertEqual(len(memory["problems"]), 1)
        problem = memory["problems"][0]
        self.assertEqual(
            (problem["id"], problem["name"], problem["opened_round"], problem["opened_run"]),
            ("P1", "rules cover too few keywords per label", 0, st.run),
        )
        first, second = problem["attempts"]
        self.assertEqual(first, {
            "run": st.run, "round": 0, "sample": "s1", "accepted": True, "backfilled": False,
            "what": "Add the keywords 'charged' and 'error', which the failing billing and bug messages use.",
            "files": ["rules.txt"],
            "predicted": ["t03", "t05"], "fixed": ["t03", "t05"], "missed": [], "regressed": [],
            "passed": 4, "total": 6, "score_before": 33.33, "score_after": 66.67,
            "domain": {"benchmark": "toy-v1", "model": "model-a"},
        })
        self.assertEqual(
            (second["round"], second["predicted"], second["fixed"], second["missed"], second["regressed"]),
            (1, ["t04", "t06"], ["t04"], ["t06"], []),
        )
        self.assertEqual(
            (second["passed"], second["total"], second["score_before"], second["score_after"]),
            (5, 6, 66.67, 83.33),
        )

    def test_branch_has_a_harness_commit_and_a_framework_commit_per_accepted_round(self):
        st = self.st
        commits = git(self.project, "log", "--reverse", "--format=%H %s", f"main..{st.branch}").splitlines()
        self.assertEqual(len(commits), 4)
        subjects = [line.split(" ", 1)[1] for line in commits]
        self.assertTrue(subjects[0].startswith(f"rsi: {st.run} r0 s1: Add the keywords"))
        self.assertEqual(subjects[1], f"rsi: {st.run} round 0 record")
        self.assertEqual(subjects[3], f"rsi: {st.run} round 1 record")
        files = lambda sha: git(self.project, "show", "--name-only", "--format=", sha).splitlines()
        self.assertEqual(files(commits[0].split()[0]), ["rules.txt"])
        self.assertEqual(sorted(files(commits[1].split()[0])), [".rsi/log.md", ".rsi/sim.json"])
        log = git(self.project, "show", f"{st.branch}:.rsi/log.md")
        self.assertIn(f"| {st.run} | 0 | 33.33 | 66.67 | s1: Add the keywords", log)
        self.assertIn("| P1 | t04 | t06 |  |", log)
        self.assertNotIn("held", log.lower())
        self.assertEqual(git(self.project, "branch", "--list", "rsi-cand/*"), "")

    def test_bad_candidates_are_dropped_before_evaluation(self):
        st = self.st
        frozen = history(st, 0, "s2/result.json")
        self.assertEqual(frozen["status"], "dropped")
        self.assertEqual(frozen["reason"], "edit outside the allowed paths: classify.py: frozen")
        self.assertIsNone(frozen["score"])
        self.assertFalse((self.rl.feedback / frozen["sha"]).exists())
        invalid = history(st, 0, "s3/result.json")
        self.assertEqual(invalid["reason"], "invalid manifest: manifest.json is not valid JSON")
        empty = history(st, 1, "s3/result.json")
        self.assertEqual(empty["reason"], "the edit changed no files")
        self.assertEqual(
            st.rounds_log[0]["dropped"],
            [{"sample": "s2", "reason": frozen["reason"]}, {"sample": "s3", "reason": invalid["reason"]}],
        )

    def test_history_keeps_every_candidate(self):
        st = self.st
        loser = history(st, 1, "s2/result.json")
        self.assertEqual(
            (loser["status"], loser["score"], loser["fixed"], loser["missed"], loser["regressed"]),
            ("evaluated", 66.67, ["t06"], [], ["t01"]),
        )
        table = {row["task"]: row["passes"] for row in loser["table"]}
        self.assertEqual(table, {"t01": False, "t02": True, "t03": True, "t04": False, "t05": True, "t06": True})
        self.assertIsNone(history(st, 1, "s3/result.json")["table"])  # dropped before evaluation
        self.assertEqual(history(st, 1, "decision.json"), {"accept": "s1", "reason": "scripted"})
        self.assertEqual(history(st, 1, "s2/manifest.json")["predicted"], ["t06"])
        diff = (self.rl.history / "round-1" / "s2" / "diff.patch").read_text()
        self.assertIn("-invoice => billing", diff)
        self.assertIn("+love => feature", diff)
        self.assertEqual(st.rounds_log[1]["accepted"], "s1")
        self.assertEqual((st.rounds_log[0]["opened"], st.rounds_log[1]["opened"]), (True, False))

    def test_later_rounds_see_the_memory(self):
        prompt = (self.rl.agents / "round-1" / "s1" / "prompt.md").read_text()
        self.assertIn('"name": "rules cover too few keywords per label"', prompt)
        self.assertIn('"fixed": [', prompt)
        self.assertIn("Harness score: 66.67", prompt)
        select = (self.rl.agents / "round-1" / "select" / "prompt.md").read_text()
        self.assertIn("## Candidate s2: 66.67 (+0.00)", select)
        self.assertIn("Regressed: t01", select)

    def test_no_heldout_data_reaches_an_agent(self):
        heldout = [json.loads(line) for line in (TOY / "bench" / "tasks.jsonl").read_text().splitlines()][6:]
        # Ids are matched as whole words: a random temp-directory name may contain "t07" by chance.
        needles = [re.compile(rf"\b{t['id']}\b") for t in heldout]
        needles += [re.compile(re.escape(t["text"])) for t in heldout]
        visible = [p for root in (self.rl.agents, self.rl.feedback, self.rl.history)
                   for p in root.rglob("*") if p.is_file()]
        self.assertGreater(len(visible), 40)
        texts = {str(p): p.read_text(errors="replace") for p in visible}
        texts["log.md"] = git(self.project, "show", f"{self.st.branch}:.rsi/log.md")
        for name, text in texts.items():
            for needle in needles:
                self.assertIsNone(needle.search(text), f"{needle.pattern} found in {name}")
        for root in (self.rl.agents, self.rl.feedback, self.rl.history):
            self.assertFalse(root.is_relative_to(Layout("toy").private))

    def test_every_agent_call_denies_private_data_and_makes_shared_folders_read_only(self):
        rule = lambda verb, path: f"{verb}(//" + str(path).lstrip("/") + "/**)"
        private = Layout("toy").private
        settings = sorted(self.rl.agents.rglob("settings.json"))
        self.assertEqual(len(settings), 10)  # 3 proposers + screen + select, in each of 2 rounds
        for path in settings:
            deny = json.loads(path.read_text())["permissions"]["deny"]
            expected = [rule("Read", private), rule("Edit", private),
                        rule("Edit", self.rl.feedback), rule("Edit", self.rl.history)]
            if path.parent.name in ("screen", "select"):  # the selector only reads the run branch's worktree
                expected.append(rule("Edit", self.rl.worktrees / "main"))
            self.assertEqual(deny, expected, str(path))


class GuardsTest(ToyCase):
    def test_tolerance_guard_overrides_the_selector(self):
        project = self.make_toy(script="script_guard.json", rounds=1)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 33.33])
        entry = st.rounds_log[0]
        self.assertIsNone(entry["accepted"])
        self.assertIn("guard: s1 scored 0.00", entry["reason"])
        self.assertEqual(entry["dropped"], [
            {"sample": "s2", "reason": "agent failed: exit code 3"},
            {"sample": "s3", "reason": "agent failed: exit code 3"},
        ])
        self.assertEqual(st.promoted, {"skipped": "no edit was accepted in this run"})
        self.assertEqual(git(project, "log", "--format=%s", f"main..{st.branch}").splitlines(),
                         [f"rsi: {st.run} round 0 record"])
        self.assertEqual(git(project, "show", f"{st.branch}:rules.txt"), "invoice => billing\ncrash => bug")
        missing = subprocess.run(["git", "cat-file", "-e", f"{st.branch}:.rsi/sim.json"], cwd=project,
                                 capture_output=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn(f"| {st.run} | 0 | 33.33 | 33.33 | none |", git(project, "show", f"{st.branch}:.rsi/log.md"))

    def test_selector_failure_keeps_candidates_at_screen_and_accepts_nothing(self):
        project = self.make_toy(script="script_selector_down.json", rounds=1, candidates=1)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 33.33])
        result = history(st, 0, "s1/result.json")
        self.assertEqual((result["status"], result["score"]), ("evaluated", 66.67))
        self.assertEqual(st.rounds_log[0]["reason"], "selector failed: exit code 3")

    def test_manifest_claims_are_checked_before_evaluation(self):
        def plan(root_cause, predicted):
            return {"edits": [{"path": "rules.txt", "append": "charged => billing\n"}],
                    "manifest": {"root_cause": root_cause, "what": "x", "predicted": predicted, "risk": ""}}
        script = self.tmp / "claims.json"
        script.write_text(json.dumps({
            "proposer": {"0": {"s1": plan({"new": "a"}, ["t01"]),
                               "s2": plan({"new": "b"}, ["t07"]),
                               "s3": plan({"id": "P9"}, ["t03"])}},
            "screen": {}, "select": {},
        }))
        project = self.make_toy(rounds=1)
        self.use_script(str(script))
        st = loop.run(project)
        reasons = {d["sample"]: d["reason"] for d in st.rounds_log[0]["dropped"]}
        self.assertEqual(reasons, {
            "s1": "predicted tasks are not failing evolve tasks: t01",
            "s2": "predicted tasks are not failing evolve tasks: t07",
            "s3": "root cause P9 is not in the memory",
        })
        self.assertEqual(st.rounds_log[0]["reason"], "no candidate reached evaluation")

    def test_failing_check_drops_a_candidate(self):
        # Inline code, because a command may not name a file the loop edits.
        check = f"{sys.executable} -c \"import sys; sys.exit(0 if 'invoice' in open('rules.txt').read() else 1)\""
        script = self.tmp / "breaks_check.json"
        script.write_text(json.dumps({
            "proposer": {"0": {"s1": {
                "edits": [{"path": "rules.txt", "write": "crash => bug\n"}],
                "manifest": {"root_cause": {"new": "a"}, "what": "x", "predicted": [], "risk": ""}}}},
            "screen": {}, "select": {},
        }))
        project = self.make_toy(rounds=1, candidates=1, check=check)
        self.use_script(str(script))
        st = loop.run(project)
        self.assertEqual(st.rounds_log[0]["dropped"],
                         [{"sample": "s1", "reason": "check failed with exit code 1"}])

    def test_a_selector_that_edits_the_worktree_changes_nothing(self):
        project = self.make_toy()
        tamper = "echo 'zzz => other' > rules.txt; echo '{}' > .rsi/sim.json; echo x > stray.txt; "
        text = (project / loop.CONFIG_PATH).read_text().replace('selector = "', 'selector = "' + tamper)
        self.assertIn(tamper, text)
        (project / loop.CONFIG_PATH).write_text(text)
        git(project, "commit", "-q", "-am", "a selector that edits files")
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33})
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual(len(memory["problems"][0]["attempts"]), 2)
        tree = git(project, "ls-tree", "-r", "--name-only", st.branch).splitlines()
        self.assertNotIn("stray.txt", tree)
        self.assertNotIn("zzz", git(project, "show", f"{st.branch}:rules.txt"))

    def test_files_the_commit_does_not_hold_are_gone_before_scoring(self):
        project = self.make_toy(rounds=1, candidates=1,
                                run_task=f"test ! -e override.log && {sys.executable} score.py {{task_file}} {{checkout}} {{out}}")
        (project / ".gitignore").write_text("*.log\n")
        git(project, "add", ".gitignore")
        git(project, "commit", "-q", "-m", "ignore logs")
        script = self.tmp / "override.json"
        script.write_text(json.dumps({
            "proposer": {"0": {"s1": {
                "edits": [{"path": "rules.txt", "append": "charged => billing\nerror => bug\n"},
                          {"path": "override.log", "write": "an ignored file a scorer might read"}],
                "manifest": {"root_cause": {"new": "a"}, "what": "x", "predicted": ["t03"], "risk": ""}}}},
            "screen": {"0": {"s1": True}}, "select": {"0": "s1"},
        }))
        self.use_script(str(script))
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67])
        self.assertEqual(history(st, 0, "s1/result.json")["files"], ["rules.txt"])

    def test_wall_clock_limit_stops_new_rounds(self):
        project = self.make_toy()
        with mock.patch.object(loop, "_out_of_time", return_value=True):
            st = loop.run(project)
        self.assertEqual((st.status, st.round, st.scores, st.rounds_log), ("done", 0, [33.33], []))
        self.assertEqual(st.heldout, {"start": 33.33, "end": 33.33})


class ResumeTest(ToyCase):
    def test_resume_after_a_kill_repeats_no_finished_rollout(self):
        call_log = self.tmp / "calls.log"
        os.environ["TOY_CALL_LOG"] = str(call_log)
        project = self.make_toy()
        real = evaluator.run_rollout
        counter = itertools.count(1)

        # 6 held-out + 6 baseline rollouts run first, so the 15th new rollout is in round 0's evaluate step.
        def dies_on_the_15th(cfg, task, rollout, checkout, out_dir, benchmark):
            if not (out_dir / evaluator.MARKER).exists() and next(counter) == 15:
                raise RuntimeError("killed")
            return real(cfg, task, rollout, checkout, out_dir, benchmark)

        with mock.patch.object(evaluator, "run_rollout", dies_on_the_15th):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        stopped = loop.find_unfinished(Layout("toy"))
        self.assertEqual((stopped.status, stopped.step, stopped.round), ("running", "evaluate", 0))

        st = loop.run(project)
        self.assertEqual(st.run, stopped.run)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33})
        ran = call_log.read_text().splitlines()
        self.assertEqual(len(ran), len(set(ran)))
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual(len(memory["problems"][0]["attempts"]), 2)

    def test_record_step_can_run_twice(self):
        project = self.make_toy(rounds=1)
        real = loop._next_round
        calls = itertools.count(1)

        def dies_after_recording(ctx):
            if next(calls) == 2:  # the first call follows baseline; the second ends round 0's record step
                raise RuntimeError("killed")
            real(ctx)

        with mock.patch.object(loop, "_next_round", dies_after_recording):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        self.assertEqual(loop.find_unfinished(Layout("toy")).step, "record")
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67])
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual(len(memory["problems"][0]["attempts"]), 1)
        log = git(project, "show", f"{st.branch}:.rsi/log.md")
        self.assertEqual(log.count(f"| {st.run} | 0 |"), 1)
        self.assertEqual(len(st.rounds_log), 1)


class KillTest(ToyCase):
    def test_resume_after_the_process_is_killed_outright(self):
        """SIGKILL gives the process no chance to clean up: no handlers, no finally blocks, a stale lock."""
        project = self.make_toy()
        call_log = self.tmp / "calls.log"
        src = Path(__file__).resolve().parents[1] / "src"
        env = dict(os.environ, TOY_CALL_LOG=str(call_log), TOY_DELAY="0.15", PYTHONPATH=str(src))
        proc = subprocess.Popen(
            [sys.executable, "-m", "rsi", "run", "--no-promote", "--project", str(project)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        ran = lambda: len(call_log.read_text().splitlines()) if call_log.exists() else 0
        while ran() < 14 and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)  # 12 rollouts score the starting harness; the 14th is inside round 0
        proc.kill()
        proc.wait()
        self.assertEqual(proc.returncode, -signal.SIGKILL)
        layout = Layout("toy")
        stopped = loop.find_unfinished(layout)
        self.assertEqual((stopped.status, stopped.round), ("running", 0))
        self.assertTrue(layout.lock_file.exists())
        time.sleep(1)  # scorers the dead process started run in their own session and finish on their own

        st = loop.run(project, promote=False)
        self.assertEqual((st.run, st.status), (stopped.run, "done"))
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33})
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual([a["round"] for a in memory["problems"][0]["attempts"]], [0, 1])
        self.assertFalse(layout.lock_file.exists())


class ContinuationTest(ToyCase):
    def test_setting_change_records_a_hop_and_carries_the_memory(self):
        project = self.make_toy()
        first = loop.run(project)
        git(project, "merge", "-q", "--ff-only", first.branch)
        (project / loop.CONFIG_PATH).write_text(self.config_text(model="model-b", rounds=1, candidates=1))
        git(project, "commit", "-q", "-am", "switch the model")
        self.use_script("script_second_run.json")

        second = loop.run(project)
        self.assertEqual(second.scores, [83.33, 100.0])
        memory = json.loads(git(project, "show", f"{second.branch}:.rsi/sim.json"))
        self.assertEqual(memory["carried_from"], [
            {"run": first.run, "benchmark": "toy-v1", "model": "model-a", "into": second.run}])
        attempts = memory["problems"][0]["attempts"]
        self.assertEqual([a["run"] for a in attempts], [first.run, first.run, second.run])
        self.assertEqual(attempts[-1]["domain"], {"benchmark": "toy-v1", "model": "model-b"})
        prompt = (Layout("toy").run(second.run).agents / "round-0" / "s1" / "prompt.md").read_text()
        self.assertIn(first.run, prompt)
        self.assertIn('"model": "model-a"', prompt)
        self.assertIn("Model that runs the tasks: model-b", prompt)

    def test_same_setting_records_no_hop(self):
        project = self.make_toy()
        first = loop.run(project)
        git(project, "merge", "-q", "--ff-only", first.branch)
        (project / loop.CONFIG_PATH).write_text(self.config_text(rounds=1, candidates=1))
        git(project, "commit", "-q", "-am", "one more round")
        self.use_script("script_second_run.json")
        second = loop.run(project)
        memory = json.loads(git(project, "show", f"{second.branch}:.rsi/sim.json"))
        self.assertEqual(memory["carried_from"], [])


class OptionsTest(ToyCase):
    def test_heldout_every_round(self):
        project = self.make_toy(heldout="every_round")
        st = loop.run(project)
        self.assertEqual(st.heldout, {"start": 33.33, "rounds": {"0": 66.67, "1": 83.33}, "end": 83.33})

    def test_live_evidence_reaches_the_prompt_but_heldout_evidence_does_not(self):
        feed = self.tmp / "feed.jsonl"
        feed.write_text("".join(json.dumps(r) + "\n" for r in [
            {"kind": "evidence", "id": "e1", "summary": "Customers keep writing charged", "task_id": "t03"},
            {"kind": "evidence", "id": "e2", "summary": "SECRET-HELDOUT-NOTE", "task_id": "t10"},
            {"kind": "task", "id": "live-1", "summary": "QUEUED-TASK-NOTE",
             "task": {"id": "t13", "text": "bill me twice", "label": "billing"}},
        ]))
        project = self.make_toy(ingest=f"cat {shlex.quote(str(feed))}", rounds=1)
        st = loop.run(project)
        rl = Layout("toy").run(st.run)
        prompt = (rl.agents / "round-0" / "s1" / "prompt.md").read_text()
        self.assertIn("(task t03): Customers keep writing charged", prompt)
        self.assertNotIn("SECRET-HELDOUT-NOTE", prompt)
        self.assertNotIn("QUEUED-TASK-NOTE", prompt)
        self.assertEqual(len((rl.history / "round-0" / "evidence.jsonl").read_text().splitlines()), 1)
        self.assertEqual(tasks.pending_count(Layout("toy")), 1)
        self.assertEqual(len(st.evolve_ids) + len(st.heldout_ids), 12)
        # The stored evidence includes the held-out note, so it must sit where agents are denied.
        stored = Layout("toy").evidence_file
        self.assertIn("SECRET-HELDOUT-NOTE", stored.read_text())
        self.assertTrue(stored.is_relative_to(Layout("toy").private))

    def test_failing_ingest_does_not_stop_the_round(self):
        project = self.make_toy(ingest="exit 9", rounds=1)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67])
