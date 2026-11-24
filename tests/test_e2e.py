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

from rsi import agents, evaluator, loop, tasks
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
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})
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
        # 3 proposers + screen + select in each of 2 rounds, plus the first try of round 0's s3, which
        # left an invalid manifest and was called once more
        self.assertEqual(len(settings), 11)
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
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})
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
        self.assertEqual(st.heldout, {"start": 33.33, "end": 33.33, "failed": {"start": 0, "final": 0, "total": 6}})


class ResumeTest(ToyCase):
    def test_resume_after_a_kill_repeats_no_finished_rollout(self):
        call_log = self.tmp / "calls.log"
        os.environ["TOY_CALL_LOG"] = str(call_log)
        project = self.make_toy()
        real = evaluator.run_rollout
        counter = itertools.count(1)

        # 6 held-out + 6 baseline rollouts run first, so the 15th new rollout is in round 0's evaluate step.
        def dies_on_the_15th(cfg, task, rollout, checkout, out_dir, benchmark, *rest):
            if not (out_dir / evaluator.MARKER).exists() and next(counter) == 15:
                raise RuntimeError("killed")
            return real(cfg, task, rollout, checkout, out_dir, benchmark, *rest)

        with mock.patch.object(evaluator, "run_rollout", dies_on_the_15th):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        stopped = loop.find_unfinished(Layout("toy"))
        self.assertEqual((stopped.status, stopped.step, stopped.round), ("running", "evaluate", 0))

        st = loop.run(project)
        self.assertEqual(st.run, stopped.run)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})
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
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual([a["round"] for a in memory["problems"][0]["attempts"]], [0, 1])
        self.assertFalse(layout.lock_file.exists())


class ExternalFilesTest(ToyCase):
    def test_nothing_of_the_loop_enters_the_repository(self):
        project = self.make_toy(external=True)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})
        self.assertTrue(st.external)
        tree = git(project, "ls-tree", "-r", "--name-only", st.branch).splitlines()
        self.assertEqual([p for p in tree if p.startswith(".rsi")], [])
        subjects = git(project, "log", "--format=%s", f"main..{st.branch}").splitlines()
        self.assertEqual(len(subjects), 2)  # the two accepted edits and nothing else
        self.assertTrue(all(s.startswith(f"rsi: {st.run} r") for s in subjects))
        memory = json.loads((self.home / "toy" / "sim.json").read_text())
        self.assertEqual(len(memory["problems"][0]["attempts"]), 2)
        self.assertEqual((self.home / "toy" / "log.md").read_text().count(f"| {st.run} |"), 2)
        self.assertEqual(git(project, "status", "--porcelain"), "")
        self.assertFalse((project / ".rsi").exists())

    def test_record_step_can_run_twice(self):
        project = self.make_toy(external=True, rounds=1)
        real = loop._next_round
        calls = itertools.count(1)

        def dies_after_recording(ctx):
            if next(calls) == 2:
                raise RuntimeError("killed")
            real(ctx)

        with mock.patch.object(loop, "_next_round", dies_after_recording):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67])
        memory = json.loads((self.home / "toy" / "sim.json").read_text())
        self.assertEqual(len(memory["problems"][0]["attempts"]), 1)
        self.assertEqual((self.home / "toy" / "log.md").read_text().count(f"| {st.run} | 0 |"), 1)

    def test_second_run_sees_the_first_runs_memory(self):
        project = self.make_toy(external=True)
        first = loop.run(project)
        self.write_config(project, external=True, rounds=1, candidates=1)  # read from disk when a run starts
        self.use_script("script_second_run.json")
        second = loop.run(project, from_ref=first.branch)
        self.assertEqual(second.scores, [83.33, 100.0])
        prompt = (Layout("toy").run(second.run).agents / "round-0" / "s1" / "prompt.md").read_text()
        self.assertIn(first.run, prompt)
        memory = json.loads((self.home / "toy" / "sim.json").read_text())
        self.assertEqual([a["run"] for a in memory["problems"][0]["attempts"]], [first.run, first.run, second.run])

    def test_a_setting_change_records_a_hop_without_a_commit(self):
        project = self.make_toy(external=True)
        first = loop.run(project)
        self.write_config(project, external=True, model="model-b", rounds=0)
        second = loop.run(project, from_ref=first.branch)
        memory = json.loads((self.home / "toy" / "sim.json").read_text())
        self.assertEqual(memory["carried_from"], [
            {"run": first.run, "benchmark": "toy-v1", "model": "model-a", "into": second.run}])
        self.assertEqual(git(project, "rev-parse", second.branch), git(project, "rev-parse", first.branch))


class AgentCallsTest(ToyCase):
    TAMPER = "(echo x > {feedback_dir}/planted.txt; echo x > {history_dir}/planted.txt) 2>/dev/null; "

    def fake_gitdir(self, marker):
        """Shell text that points the agent's worktree at a git directory of its own, whose config runs a command."""
        return ("mkdir -p fakegit/objects fakegit/refs; echo 'ref: refs/heads/main' > fakegit/HEAD; "
                f"printf '[core]\\n\\trepositoryformatversion = 0\\n\\tfsmonitor = \"touch {marker}\"\\n' > fakegit/config; "
                "rm -f .git; echo 'gitdir: fakegit' > .git; ")

    def test_a_proposer_cannot_make_the_loop_run_its_git_config(self):
        marker = self.tmp / "PWNED"
        project = self.make_toy(rounds=1, candidates=1, proposer_prefix=self.fake_gitdir(marker))
        st = loop.run(project)
        self.assertFalse(marker.exists(), "the loop ran a command from a git config an agent wrote")
        self.assertEqual(st.rounds_log[0]["dropped"],
                         [{"sample": "s1", "reason": "agent failed: the agent changed the worktree's .git"}])

    def test_a_selector_cannot_make_the_loop_run_its_git_config(self):
        marker = self.tmp / "PWNED"
        project = self.make_toy(rounds=1, candidates=1, selector_prefix=self.fake_gitdir(marker))
        st = loop.run(project)
        self.assertFalse(marker.exists(), "the loop ran a command from a git config an agent wrote")
        self.assertEqual(st.rounds_log[0]["reason"], "selector failed: the agent changed the worktree's .git")
        self.assertEqual(st.status, "done")

    def test_shared_folders_are_read_only_while_agents_run(self):
        project = self.make_toy(proposer_prefix=self.TAMPER, selector_prefix=self.TAMPER)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        rl = Layout("toy").run(st.run)
        self.assertEqual(list(rl.feedback.rglob("planted.txt")) + list(rl.history.rglob("planted.txt")), [])
        (rl.feedback / "writable-again.txt").write_text("x")

    def test_resume_restores_write_access(self):
        project = self.make_toy()
        with mock.patch.object(agents, "run_agent", side_effect=RuntimeError("killed")):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        stopped = loop.find_unfinished(Layout("toy"))
        self.assertEqual(stopped.step, "propose")
        rl = Layout("toy").run(stopped.run)
        # A process killed outright leaves the folders as they were while the agents ran.
        agents.set_writable([rl.feedback, rl.history], False)
        self.assertFalse(os.access(rl.feedback, os.W_OK))
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])

    def test_an_agent_that_leaves_no_output_is_called_once_more(self):
        marks = self.tmp / "marks"
        marks.mkdir()
        first_call_does_nothing = (f"m={shlex.quote(str(marks))}/$RSI_ROUND-$RSI_SAMPLE; "
                                   'if [ ! -e "$m" ]; then touch "$m"; exit 0; fi; ')
        project = self.make_toy(proposer_prefix=first_call_does_nothing)
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        agents_dir = Layout("toy").run(st.run).agents / "round-0"
        self.assertTrue((agents_dir / "s1.try1" / "prompt.md").is_file())
        self.assertFalse((agents_dir / "s1.try1" / "manifest.json").exists())
        self.assertTrue((agents_dir / "s1" / "manifest.json").is_file())

    def test_a_selector_that_fails_twice_accepts_nothing(self):
        project = self.make_toy(script="script_selector_down.json", rounds=1, candidates=1)
        st = loop.run(project)
        self.assertEqual(st.rounds_log[0]["reason"], "selector failed: exit code 3")
        agents_dir = Layout("toy").run(st.run).agents / "round-0"
        self.assertTrue((agents_dir / "select.try1").is_dir())
        self.assertTrue((agents_dir / "screen.try1").is_dir())

    def test_a_timeout_is_not_tried_again(self):
        project = self.make_toy(rounds=1, candidates=1, agent_timeout_s=1, proposer_prefix="sleep 30; ")
        st = loop.run(project)
        self.assertEqual(st.rounds_log[0]["dropped"], [{"sample": "s1", "reason": "agent failed: timed out after 1s"}])
        self.assertFalse((Layout("toy").run(st.run).agents / "round-0" / "s1.try1").exists())

    def test_output_written_before_a_denied_action_counts(self):
        denied = "; echo '{\"denied_actions\": [{\"action\": \"read_file\"}]}' > {out_dir}/agent-result.json"
        project = self.make_toy(rounds=1, candidates=1)
        text = (project / loop.CONFIG_PATH).read_text()
        patched = text.replace('--out {out_dir}"\nselector', '--out {out_dir}' + denied.replace('"', '\\"') + '"\nselector', 1)
        self.assertNotEqual(text, patched)
        (project / loop.CONFIG_PATH).write_text(patched)
        git(project, "commit", "-q", "-am", "a proposer that is denied after writing its manifest")
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67])
        self.assertFalse((Layout("toy").run(st.run).agents / "round-0" / "s1.try1").exists())


class PairedTest(ToyCase):
    def conditions(self, factors=None, fail=None, where="/main"):
        """Change the scoring conditions before chosen steps.

        `factors` maps (step, round) to what every score is multiplied by from that step on, and `fail`
        names the (step, round) in which rollouts of the run branch's own worktree fail.
        """
        factor_file, fail_file = self.tmp / "factor", self.tmp / "fail"
        os.environ["TOY_FACTOR_FILE"], os.environ["TOY_FAIL_FILE"] = str(factor_file), str(fail_file)
        factors = factors or {}

        def before(step, real):
            def handler(ctx):
                key = (step, ctx.state.round)
                if key in factors:
                    factor_file.write_text(str(factors[key]))
                if key == fail:
                    fail_file.write_text(where)
                else:
                    fail_file.unlink(missing_ok=True)
                real(ctx)
            return handler

        steps = ("baseline", "evaluate", "confirm", "heldout_end")
        self.enterContext(mock.patch.dict(loop._HANDLERS, {s: before(s, loop._HANDLERS[s]) for s in steps
                                                          if s in loop._HANDLERS}))

    def test_a_drop_in_the_machines_is_not_a_loss(self):
        project = self.make_toy(tasks_extra="paired = true", rounds=1, candidates=1)
        self.conditions({("baseline", 0): 1.0, ("evaluate", 0): 0.5, ("heldout_end", 1): 1.0})
        st = loop.run(project)
        entry = st.rounds_log[0]
        self.assertEqual((entry["accepted"], entry["score_before"], entry["score_after"]), ("s1", 16.67, 33.33))
        self.assertEqual(st.scores, [33.33, 33.33])
        rl = Layout("toy").run(st.run)
        self.assertTrue((rl.feedback / f"{st.start_sha}.r0" / "t01" / "r1" / "rollout.json").is_file())
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        attempt = memory["problems"][0]["attempts"][0]
        self.assertEqual((attempt["score_before"], attempt["score_after"], attempt["fixed"]),
                         (16.67, 33.33, ["t03", "t05"]))
        self.assertEqual((st.heldout["start"], st.heldout["end"]), (33.33, 66.67))

    def test_pairing_decides_whether_a_real_gain_is_seen(self):
        for paired, expected in ((False, None), (True, "s1")):
            with self.subTest(paired=paired):
                tmp = self.tmp / f"paired-{paired}"
                self.tmp, self.home = tmp, tmp / "home"
                os.environ["RSI_HOME"] = str(self.home)
                project = self.make_toy(tasks_extra=f"paired = {str(paired).lower()}", rounds=1, candidates=1,
                                        budget_extra="min_gain = 1")
                self.conditions({("baseline", 0): 1.0, ("evaluate", 0): 0.5, ("heldout_end", 1): 1.0})
                st = loop.run(project)
                self.assertEqual(st.rounds_log[0]["accepted"], expected)
                if not paired:  # the candidate's 33.33 is compared with a baseline measured before the machines slowed
                    self.assertTrue(st.rounds_log[0]["reason"].startswith(
                        "guard: s1 gained 0.00 points, less than the 1.00 required"), st.rounds_log[0]["reason"])

    def test_every_scoring_of_the_current_version_is_marked(self):
        log = self.tmp / "current.log"
        run_task = (f"echo \"$RSI_CURRENT $PWD\" >> {shlex.quote(str(log))}; "
                    f"{sys.executable} score.py {{task_file}} {{checkout}} {{out}}")
        project = self.make_toy(tasks_extra="paired = true", rounds=1, candidates=1, run_task=run_task)
        st = loop.run(project)
        self.assertEqual(st.rounds_log[0]["accepted"], "s1")
        marks = {}
        for line in log.read_text().splitlines():
            flag, _, where = line.partition(" ")
            marks.setdefault(Path(where).name, set()).add(flag)
        self.assertEqual(marks, {"main": {"1"}, "start": {"1"}, "r0-s1": {""}})

    def test_a_gain_that_holds_is_kept_with_the_second_measurement(self):
        project = self.make_toy(tasks_extra="paired = true\nconfirm = true", rounds=1, candidates=1)
        self.conditions({("baseline", 0): 1.0, ("evaluate", 0): 1.0, ("confirm", 0): "0.9 /r0-s1"})
        st = loop.run(project)
        entry = st.rounds_log[0]
        self.assertEqual(entry["accepted"], "s1")
        self.assertEqual((entry["score_before"], entry["score_after"]), (33.33, 60.0))  # from the second measurement
        rl = Layout("toy").run(st.run)
        self.assertTrue((rl.feedback / f"{st.start_sha}.c0").is_dir())
        self.assertTrue((rl.feedback / f"{entry['sha']}.c0").is_dir())
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        self.assertEqual(memory["problems"][0]["attempts"][0]["score_after"], 60.0)

    def test_a_gain_that_does_not_hold_is_not_kept(self):
        project = self.make_toy(tasks_extra="paired = true\nconfirm = true", rounds=1, candidates=1,
                                budget_extra="min_gain = 10")
        self.conditions({("baseline", 0): 1.0, ("evaluate", 0): 1.0, ("confirm", 0): "0.5 /r0-s1"})
        st = loop.run(project)
        entry = st.rounds_log[0]
        self.assertIsNone(entry["accepted"])
        self.assertTrue(entry["reason"].startswith(
            "guard: s1 did not hold up when measured again: it gained 0.00 points, less than the 10.00 required"),
            entry["reason"])
        self.assertEqual(st.scores, [33.33, 33.33])

    def test_the_next_round_reads_the_newest_results(self):
        project = self.make_toy(tasks_extra="paired = true")
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        rl = Layout("toy").run(st.run)
        prompt = (rl.agents / "round-1" / "s1" / "prompt.md").read_text()
        accepted_in_round_0 = history(st, 0, "s1/result.json")["sha"]
        self.assertIn(f"{rl.feedback}/{accepted_in_round_0}/", prompt)
        # round 1 measured that harness again, beside its candidates
        self.assertTrue((rl.feedback / f"{accepted_in_round_0}.r1").is_dir())

    def test_heldout_start_and_end_are_scored_together(self):
        project = self.make_toy(tasks_extra="paired = true", metrics=True)
        st = loop.run(project)
        self.assertEqual((st.heldout["start"], st.heldout["end"]), (33.33, 83.33))
        self.assertEqual(st.heldout["metrics_start"], {"rules": 2.0, "labels": 2.0})
        self.assertEqual(st.heldout["metrics_end"], {"rules": 5.0, "labels": 3.0})
        rl = Layout("toy").run(st.run)
        self.assertEqual(sorted(p.name for p in rl.heldout.iterdir()), ["end-final", "end-start"])
        self.assertFalse((rl.worktrees / "start").exists())
        self.assertEqual(git(project, "worktree", "list").count("\n"), 0)  # only the owner's checkout is left

    def test_failed_heldout_runs_are_reported_and_make_drafts(self):
        from rsi import promote
        project = self.make_toy(tasks_extra="paired = true", rounds=1, candidates=1, metrics=True,
                                tail='[[constraints]]\nmetric = "rules"\nmax_increase_pct = 200')
        self.conditions(fail=("heldout_end", 1), where="/start")
        st = loop.run(project, promote=False)
        self.assertEqual(st.heldout["failed"], {"start": 6, "final": 0, "total": 6})
        self.assertEqual(promote.run_warning(st),
                         "the held-out measurement had failed runs (start 6 of 6, final 0 of 6), so its numbers are not reliable")
        from rsi import cli
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.main(["status", "--project", str(project)])
        self.assertIn("Held-out: failed runs: start 6 of 6, final 0 of 6", out.getvalue())

    def test_failed_heldout_runs_are_reported_without_pairing(self):
        project = self.make_toy(rounds=1, candidates=1)
        self.conditions(fail=("heldout_end", 1))
        st = loop.run(project, promote=False)
        self.assertEqual(st.heldout["failed"], {"start": 0, "final": 6, "total": 6})

    def test_nothing_accepted_scores_heldout_once(self):
        project = self.make_toy(script="script_guard.json", tasks_extra="paired = true", rounds=1, candidates=1)
        st = loop.run(project)
        self.assertEqual((st.heldout["start"], st.heldout["end"]), (33.33, 33.33))
        self.assertEqual([p.name for p in Layout("toy").run(st.run).heldout.iterdir()], ["end-final"])

    def test_a_failed_baseline_run_stops_the_round(self):
        project = self.make_toy(tasks_extra="paired = true", rounds=1, candidates=1)
        self.conditions(fail=("evaluate", 0))
        st = loop.run(project)
        entry = st.rounds_log[0]
        self.assertIsNone(entry["accepted"])
        self.assertEqual(entry["reason"], "the round's baseline had failed runs; nothing can be compared")
        self.assertFalse((Layout("toy").run(st.run).agents / "round-0" / "select").exists())
        self.assertEqual(st.scores, [33.33, 33.33])  # the failed measurement is not taken as the harness's score
        self.assertEqual(st.eval_dir, "")

    def test_resume_inside_a_paired_round(self):
        call_log = self.tmp / "calls.log"
        os.environ["TOY_CALL_LOG"] = str(call_log)
        project = self.make_toy(tasks_extra="paired = true")
        real = evaluator.run_rollout
        counter = itertools.count(1)

        # 6 baseline rollouts run first; the 9th new rollout is inside round 0's batch of 12.
        def dies_on_the_9th(cfg, task, rollout, checkout, out_dir, benchmark, *rest):
            if not (out_dir / evaluator.MARKER).exists() and next(counter) == 9:
                raise RuntimeError("killed")
            return real(cfg, task, rollout, checkout, out_dir, benchmark, *rest)

        with mock.patch.object(evaluator, "run_rollout", dies_on_the_9th):
            with self.assertRaises(RuntimeError):
                loop.run(project)
        self.assertEqual(loop.find_unfinished(Layout("toy")).step, "evaluate")
        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        self.assertEqual((st.heldout["start"], st.heldout["end"]), (33.33, 83.33))
        ran = call_log.read_text().splitlines()
        self.assertEqual(len(ran), len(set(ran)))


class LimitsTest(ToyCase):
    """The toy with metrics: `rules` (rule lines, 2 at the start) and `labels` (distinct labels, 2)."""

    @staticmethod
    def limit(metric, **bound):
        (key, value), = bound.items()
        return f'[[constraints]]\nmetric = "{metric}"\n{key} = {value}'

    def one_candidate(self, edits, **manifest):
        path = self.tmp / "one.json"
        plan = {"edits": edits, "manifest": {"root_cause": {"new": "a"}, "what": "An edit.", "predicted": [],
                                             "risk": "none", **manifest}}
        path.write_text(json.dumps({"proposer": {"0": {"s1": plan}}, "screen": {"0": {"s1": True}},
                                    "select": {"0": "s1"}}))
        self.use_script(str(path))

    def test_a_limit_overrides_the_selector(self):
        project = self.make_toy(metrics=True, rounds=1, candidates=1, tail=self.limit("rules", max_increase_pct=60))
        st = loop.run(project)
        entry = st.rounds_log[0]
        self.assertIsNone(entry["accepted"])
        self.assertTrue(entry["reason"].startswith("guard: s1 is past a limit: rules +100.00% of +60.00% allowed"),
                        entry["reason"])
        result = history(st, 0, "s1/result.json")
        self.assertEqual(result["limits"][0]["state"], "over")
        self.assertEqual(result["metrics"], {"rules": 4.0, "labels": 2.0})
        self.assertEqual(result["observed"], {"rules": 100.0, "labels": 0.0})
        self.assertEqual(st.metric_ratio, {})

    def test_a_limit_is_cumulative_across_rounds(self):
        project = self.make_toy(metrics=True, tail=self.limit("rules", max_increase_pct=120))
        st = loop.run(project)
        self.assertEqual([e["accepted"] for e in st.rounds_log], ["s1", None])
        self.assertTrue(st.rounds_log[1]["reason"].startswith(
            "guard: s1 is past a limit: rules +150.00% of +120.00% allowed"), st.rounds_log[1]["reason"])
        self.assertEqual(st.metric_ratio, {"rules": 2.0})
        prompt = (Layout("toy").run(st.run).agents / "round-1" / "s1" / "prompt.md").read_text()
        self.assertIn("- rules may rise at most 120.00% in this run; +100.00% used", prompt)
        self.assertIn("- rules: 4.00", prompt)

    def test_a_limit_of_zero(self):
        zero = self.limit("labels", max_decrease_pct=0)
        project = self.make_toy(metrics=True, rounds=1, candidates=1, tail=zero)
        self.one_candidate([{"path": "rules.txt", "write": "crash => bug\nerror => bug\n"}])  # labels 2 -> 1
        st = loop.run(project)
        self.assertTrue(st.rounds_log[0]["reason"].startswith(
            "guard: s1 is past a limit: labels -50.00% of -0.00% allowed"), st.rounds_log[0]["reason"])
        self.assertIsNone(st.rounds_log[0]["accepted"])

    def test_a_limit_of_zero_allows_no_change(self):
        project = self.make_toy(metrics=True, rounds=1, candidates=1, tail=self.limit("labels", max_decrease_pct=0))
        st = loop.run(project)  # script.json: labels stay at 2
        self.assertEqual(st.rounds_log[0]["accepted"], "s1")

    def test_minimum_gain(self):
        project = self.make_toy(rounds=1, candidates=1, budget_extra="min_gain = 40")
        st = loop.run(project)
        self.assertIsNone(st.rounds_log[0]["accepted"])
        self.assertTrue(st.rounds_log[0]["reason"].startswith(
            "guard: s1 gained 33.34 points, less than the 40.00 required"), st.rounds_log[0]["reason"])
        project_ok = None  # a gain at the minimum is enough
        self.write_config(self.tmp / "project", rounds=1, candidates=1, budget_extra="min_gain = 33.34")
        git(self.tmp / "project", "commit", "-q", "-am", "lower the minimum gain")
        st = loop.run(self.tmp / "project")
        self.assertEqual(st.rounds_log[0]["accepted"], "s1")

    def test_expected_and_observed_reach_the_memory(self):
        project = self.make_toy(metrics=True, rounds=1, candidates=1)
        self.one_candidate([{"path": "rules.txt", "append": "charged => billing\nerror => bug\n"}],
                           title="Add two keywords", expected={"rules": 100})
        st = loop.run(project)
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        attempt = memory["problems"][0]["attempts"][0]
        self.assertEqual(attempt["expected"], {"rules": 100.0})
        self.assertEqual(attempt["observed"], {"rules": 100.0, "labels": 0.0})
        self.assertEqual(attempt["metrics_before"], {"rules": 2.0, "labels": 2.0})
        self.assertEqual(attempt["metrics_after"], {"rules": 4.0, "labels": 2.0})
        entry = st.rounds_log[0]
        self.assertEqual((entry["title"], entry["risk"], entry["runs"]), ("Add two keywords", "none", 1))
        self.assertEqual(entry["sha"], git(project, "rev-parse", f"{st.branch}~1"))
        self.assertEqual(entry["observed"], {"rules": 100.0, "labels": 0.0})
        log = git(project, "show", f"{st.branch}:.rsi/log.md")
        self.assertIn("| s1: An edit. (rules +100.00%, labels +0.00%) |", log)

    def test_a_project_without_metrics_keeps_its_memory_format(self):
        project = self.make_toy(rounds=1, candidates=1)
        st = loop.run(project)
        memory = json.loads(git(project, "show", f"{st.branch}:.rsi/sim.json"))
        attempt = memory["problems"][0]["attempts"][0]
        for key in ("expected", "observed", "metrics_before", "metrics_after"):
            self.assertNotIn(key, attempt)

    def test_heldout_limits_in_paired_mode(self):
        project = self.make_toy(metrics=True, tasks_extra="paired = true",
                                tail=self.limit("rules", max_increase_pct=200))
        st = loop.run(project)
        self.assertEqual([e["accepted"] for e in st.rounds_log], ["s1", "s1"])
        status = st.heldout["limits"][0]
        self.assertEqual((status["metric"], status["total_pct"], status["state"]), ("rules", 150.0, "ok"))


class PerEditE2ETest(ToyCase):
    def test_a_run_ends_with_one_pull_request_per_accepted_edit(self):
        from tests.test_promote import FAKE_GH_STATEFUL
        project = self.make_toy(external=True, metrics=True, tasks_extra="paired = true",
                                tail='[promotion]\nmode = "per_edit"\ndraft = true')
        remote = self.tmp / "remote.git"
        git(self.tmp, "init", "-q", "--bare", str(remote))
        git(project, "remote", "add", "origin", str(remote))
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir / "gh").write_text(FAKE_GH_STATEFUL)
        (bin_dir / "gh").chmod(0o755)
        prs = self.tmp / "prs.json"
        os.environ.update(PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", GH_PRS=str(prs),
                          GH_CALLS=str(self.tmp / "gh-calls.jsonl"))

        st = loop.run(project)
        self.assertEqual(st.scores, [33.33, 66.67, 83.33])
        records = st.promoted["prs"]
        self.assertEqual([(r["round"], r["branch"], r["stacked"], r["draft"]) for r in records],
                         [(0, f"rsi/{st.run}-r0", False, True), (1, f"rsi/{st.run}-r1", True, True)])
        self.assertTrue(all(r["url"].startswith("https://example.test/pull/") for r in records))
        # round 0's branch carries that edit alone; round 1's edit does not apply without it, so its branch has both
        first = git(remote, "show", f"rsi/{st.run}-r0:rules.txt").splitlines()
        self.assertEqual(first, ["invoice => billing", "crash => bug", "charged => billing", "error => bug"])
        self.assertEqual(git(remote, "rev-parse", f"rsi/{st.run}-r1"), git(project, "rev-parse", st.branch))
        self.assertEqual(git(remote, "ls-tree", "-r", "--name-only", f"rsi/{st.run}-r1").count(".rsi"), 0)
        created = json.loads(prs.read_text())
        self.assertEqual([p["base"] for p in created], ["main", "main"])
        self.assertTrue(created[0]["title"].startswith("Add the keywords"))
        self.assertIn("| rules | 2.00 | 4.00 | +100.00% |", created[0]["body"])
        self.assertIn("Each side: 1 run per task, measured in the same batch in round 0.", created[0]["body"])
        self.assertIn("This branch contains those edits. Merge them first.", created[1]["body"])
        self.assertIn("Held-out: 33.33 → 83.33", created[1]["body"])
        self.assertEqual(git(project, "worktree", "list").count("\n"), 0)

        from rsi import cli
        import contextlib, io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(["status", "--project", str(project)]), 0)
            self.assertEqual(cli.main(["promote", "--project", str(project)]), 0)
        self.assertIn(f"  round 0, rsi/{st.run}-r0: https://example.test/pull/1 (draft)", out.getvalue())
        self.assertIn(f"  round 1, rsi/{st.run}-r1; contains the earlier edits: https://example.test/pull/2 (draft)",
                      out.getvalue())
        self.assertEqual(len(json.loads(prs.read_text())), 2)  # promoting again opened nothing new


class ProgressTest(ToyCase):
    def run_capturing(self, project, tty=False, **kwargs):
        """loop.run with stderr captured; returns (state or the exception, text)."""
        from tests.test_progress import FakeStream
        stream = FakeStream(tty)
        with mock.patch("sys.stderr", stream):
            try:
                st = loop.run(project, **kwargs)
            except loop.RunError as error:
                st = error
        return st, stream.getvalue()

    def test_lines_for_steps_and_events(self):
        project = self.make_toy(rounds=1)
        st, text = self.run_capturing(project, promote=False)
        self.assertEqual(st.scores, [33.33, 66.67])
        self.assertIn(f"[rsi] {st.run} round 1/1: measuring", text)              # step lines, rounds counted from 1
        self.assertIn(f"[rsi] {st.run} round 0/1: baseline", text)
        self.assertIn("[rsi] round 1/1: accepted s1, evolve 33.33 -> 66.67", text)
        self.assertNotRegex(text, r"\dh\d\dm total")                          # a short run: elapsed time stays small
        self.assertRegex(text, r"\[rsi\] round 0/1 · starting · .* 0s total")
        self.assertIn("[rsi] round 1 · measuring · s1 · t03 run 1/1: score 1.00 in", text)
        self.assertIn("[rsi] round 1 · proposer s1: done in", text)
        self.assertIn("[rsi] round 1 · selector (select): done in", text)
        self.assertIn("[rsi] round 1/1 · measuring · runs 0/6, 0 failed", text)  # the live line, printed at step start
        self.assertIn("[rsi] round 0/1 · baseline · runs 0/6", text)
        self.assertNotIn("\r", text)

    def test_failures_only(self):
        fail_file = self.tmp / "fail"
        fail_file.write_text("/r0-s1")
        os.environ["TOY_FAIL_FILE"] = str(fail_file)
        project = self.make_toy(rounds=1, tail='[progress]\nevents = "failures"')
        st, text = self.run_capturing(project, promote=False)
        self.assertIn("· s1 · t01 run 1/1 FAILED: exit code 7", text)
        self.assertNotIn(": score 1.00 in", text)
        self.assertNotIn("proposer s1: done in", text)
        self.assertIn("[rsi] round 1 · proposer s3: failed", text)              # s3 left an invalid manifest

    def test_a_custom_template(self):
        project = self.make_toy(rounds=1, tail='[progress]\nline = "R{round} {step} {done}/{total} err={errors} kept={accepted}"')
        st, text = self.run_capturing(project, promote=False)
        self.assertIn("[rsi] R1 measuring 0/6 err=", text)
        self.assertIn("kept=1", text)

    def test_resumed_runs_count_without_lines(self):
        project = self.make_toy(rounds=1)
        real = evaluator.run_rollout
        counter = itertools.count(1)

        def dies_on_the_15th(cfg, task, rollout, checkout, out_dir, benchmark, *rest):
            if not (out_dir / evaluator.MARKER).exists() and next(counter) == 15:
                raise RuntimeError("killed")
            return real(cfg, task, rollout, checkout, out_dir, benchmark, *rest)

        with mock.patch.object(evaluator, "run_rollout", dies_on_the_15th):
            with self.assertRaises(RuntimeError):
                self.run_capturing(project, promote=False)
        stopped = loop.find_unfinished(Layout("toy"))
        sha = next(c["sha"] for c in stopped.candidates if c["sample"] == "s1")
        finished = len(list((Layout("toy").run(stopped.run).feedback / sha).rglob(evaluator.MARKER)))
        self.assertTrue(1 <= finished < 6, finished)
        st, text = self.run_capturing(project, promote=False)
        self.assertEqual(st.scores, [33.33, 66.67])
        measured = [line for line in text.splitlines() if line.startswith("[rsi] round 1 · measuring · s1 ·")]
        self.assertEqual(len(measured), 6 - finished)          # runs finished before the kill print no line

    def test_a_failed_step_leaves_a_clean_terminal(self):
        import threading
        project = self.make_toy(run_task="exit 7")
        error, text = self.run_capturing(project, tty=True)
        self.assertIsInstance(error, loop.RunError)
        self.assertTrue(text.endswith("\r\x1b[K"), repr(text[-40:]))
        self.assertNotIn("rsi-progress", [t.name for t in threading.enumerate()])


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
        self.assertEqual(st.heldout, {"start": 33.33, "rounds": {"0": 66.67, "1": 83.33}, "end": 83.33, "failed": {"start": 0, "final": 0, "total": 6}})

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
