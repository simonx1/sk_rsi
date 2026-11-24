import os
import time
import unittest
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

from rsi import config, loop, workspace
from rsi.state import Layout, LockError, RunState
from tests.helpers import git
from tests.toy import ToyCase


class PlanTest(unittest.TestCase):
    def test_planned_task_runs(self):
        self.assertEqual(loop.planned_task_runs(10, 10, 2, 5, 3, every_round=False), 360)
        self.assertEqual(loop.planned_task_runs(10, 10, 2, 5, 3, every_round=True), 440)
        self.assertEqual(loop.planned_task_runs(6, 6, 1, 2, 3, every_round=False), 54)
        self.assertEqual(loop.planned_task_runs(6, 6, 1, 0, 3, every_round=False), 18)

    def test_planned_task_runs_paired(self):
        self.assertEqual(loop.planned_task_runs(1, 1, 4, 5, 3, every_round=False, paired=True), 92)
        self.assertEqual(loop.planned_task_runs(1, 1, 4, 5, 3, every_round=False, paired=True, confirm=True), 132)
        self.assertEqual(loop.planned_task_runs(6, 6, 1, 2, 3, every_round=False, paired=True), 66)

    def test_agent_steps_left(self):
        self.assertEqual(loop._agent_steps_left("baseline", 0, 5), 15)
        self.assertEqual(loop._agent_steps_left("ingest", 0, 5), 15)
        self.assertEqual(loop._agent_steps_left("propose", 0, 5), 14)      # this one is under way
        self.assertEqual(loop._agent_steps_left("screen", 2, 5), 7)
        self.assertEqual(loop._agent_steps_left("evaluate", 2, 5), 7)
        self.assertEqual(loop._agent_steps_left("record", 4, 5), 0)
        self.assertEqual(loop._agent_steps_left("heldout_end", 5, 5), 0)

    def test_tolerance_is_compared_at_score_precision(self):
        # 0.07 - 0.01 is 0.060000000000000005 in floating point; a candidate at 0.06 is exactly at the limit.
        self.assertFalse(loop._below_tolerance(current=0.07, candidate=0.06, tolerance=0.01))
        self.assertFalse(loop._below_tolerance(current=66.67, candidate=65.67, tolerance=1.0))
        self.assertTrue(loop._below_tolerance(current=66.67, candidate=65.66, tolerance=1.0))
        self.assertFalse(loop._below_tolerance(current=33.33, candidate=100.0, tolerance=0.0))

    def test_the_running_step_counts_toward_the_wall_clock_limit(self):
        def ctx(spent, running_for):
            return SimpleNamespace(state=SimpleNamespace(active_seconds=spent),
                                   cfg=SimpleNamespace(max_wall_clock_h=1.0),
                                   step_began=time.monotonic() - running_for)
        self.assertFalse(loop._out_of_time(ctx(spent=3590.0, running_for=2.0)))
        self.assertTrue(loop._out_of_time(ctx(spent=3599.0, running_for=2.0)))
        self.assertTrue(loop._out_of_time(ctx(spent=3601.0, running_for=0.0)))


class ValidateTest(ToyCase):
    def setUp(self):
        super().setUp()
        self.project = self.make_toy()
        self.cfg = config.load(self.project / loop.CONFIG_PATH)
        self.layout = Layout("toy")

    def test_toy_is_valid(self):
        v = loop.validate(self.project, self.cfg, self.layout)
        self.assertEqual(v.errors, [])
        self.assertEqual((len(v.evolve), len(v.heldout)), (6, 6))
        self.assertEqual(len(v.warnings), 1)
        self.assertIn("custom agent command", v.warnings[0])

    def test_default_agents_give_no_warning(self):
        cfg = replace(self.cfg, proposer=None, selector=None)
        self.assertEqual(loop.validate(self.project, cfg, self.layout).warnings, [])

    def test_task_file_inside_the_repo(self):
        inside = self.project / "bench.jsonl"
        inside.write_text('{"id": "a"}\n{"id": "b"}\n')
        v = loop.validate(self.project, replace(self.cfg, tasks_file=str(inside)), self.layout)
        self.assertEqual(len(v.errors), 1)
        self.assertIn("inside the repo", v.errors[0])

    def test_missing_task_file(self):
        v = loop.validate(self.project, replace(self.cfg, tasks_file=str(self.tmp / "nope.jsonl")), self.layout)
        self.assertIn("task file not found", v.errors[0])

    def test_both_sets_need_a_task(self):
        only = self.tmp / "one.jsonl"
        only.write_text('{"id": "solo"}\n')
        v = loop.validate(self.project, replace(self.cfg, tasks_file=str(only)), self.layout)
        self.assertIn("both sets need at least one task", v.errors[0])

    def test_an_empty_task_file_is_an_error(self):
        blank = self.tmp / "blank.jsonl"
        blank.write_text("\n\n")
        v = loop.validate(self.project, replace(self.cfg, tasks_file=str(blank)), self.layout)
        self.assertIn("both sets need at least one task", v.errors[0])

    def test_a_path_with_glob_characters_cannot_be_denied_reliably(self):
        odd = self.tmp / "bench[2026]" / "tasks.jsonl"
        odd.parent.mkdir()
        odd.write_text('{"id": "a"}\n{"id": "b"}\n')
        v = loop.validate(self.project, replace(self.cfg, tasks_file=str(odd)), self.layout)
        self.assertEqual(len(v.errors), 1)
        self.assertIn("pattern characters", v.errors[0])

    def test_command_the_loop_could_rewrite(self):
        cfg = replace(self.cfg, editable=("**",), frozen=(), run_task="score.py {task_file} {checkout} {out}")
        v = loop.validate(self.project, cfg, self.layout)
        self.assertEqual(v.errors, [
            "commands.run_task names a file the loop may edit; add it to harness.frozen, "
            "or call it from a frozen script"])

    def test_agents_are_denied_the_private_directory_and_a_task_file_kept_elsewhere(self):
        self.assertEqual(loop.denied_paths(self.cfg, self.layout), [self.layout.private])
        elsewhere = self.tmp / "bench" / "tasks.jsonl"
        cfg = replace(self.cfg, tasks_file=str(elsewhere))
        self.assertEqual(loop.denied_paths(cfg, self.layout), [self.layout.private, elsewhere])

    def test_run_check(self):
        self.assertIsNone(loop.run_check(self.cfg, self.project))
        self.assertIsNone(loop.run_check(replace(self.cfg, check=None), self.project))
        problem = loop.run_check(replace(self.cfg, check="echo nope; exit 3"), self.project)
        self.assertEqual(problem, "check failed with exit code 3: nope")


class StartTest(ToyCase):
    def test_zero_rounds_scores_both_sets_and_finishes(self):
        project = self.make_toy()
        st = loop.run(project, rounds=0)
        self.assertEqual((st.status, st.step, st.round), ("done", "done", 0))
        self.assertEqual(st.scores, [33.33])
        self.assertEqual(st.heldout, {"start": 33.33, "end": 33.33, "failed": {"start": 0, "final": 0, "total": 6}})
        self.assertEqual(st.promoted, {"skipped": "no edit was accepted in this run"})
        self.assertEqual((st.benchmark, st.base_ref), ("toy-v1", "main"))
        self.assertEqual(st.evolve_ids, ["t01", "t02", "t03", "t04", "t05", "t06"])
        self.assertTrue(workspace.branch_exists(project, st.branch))
        rl = Layout("toy").run(st.run)
        self.assertFalse((rl.worktrees / "main").exists())
        self.assertEqual(RunState.load(rl.state_file), st)

    def test_heldout_results_stay_private(self):
        project = self.make_toy()
        st = loop.run(project, rounds=0)
        rl = Layout("toy").run(st.run)
        self.assertTrue((rl.heldout / st.eval_sha / "t07" / "r1" / "rollout.json").is_file())
        self.assertTrue((rl.feedback / st.eval_sha / "t01" / "r1" / "rollout.json").is_file())
        self.assertFalse((rl.feedback / st.eval_sha / "t07").exists())
        self.assertTrue(rl.heldout.is_relative_to(self.home / "toy" / "private"))
        self.assertTrue(rl.state_file.is_relative_to(self.home / "toy" / "private"))

    def test_config_must_be_committed_on_base(self):
        project = self.make_toy()
        git(project, "rm", "-q", "--cached", ".rsi/config.toml")
        git(project, "commit", "-q", "-m", "untrack the config")
        with self.assertRaises(loop.RunError) as caught:
            loop.run(project, rounds=0)
        self.assertEqual(str(caught.exception), ".rsi/config.toml is not committed on main; commit it there first")

    def test_malformed_config_is_a_config_error(self):
        project = self.make_toy()
        (project / loop.CONFIG_PATH).write_text("[project\n")
        with self.assertRaises(config.ConfigError):
            loop.run(project, rounds=0)
        (project / loop.CONFIG_PATH).write_text(self.config_text())
        git(project, "checkout", "-q", "-b", "broken")
        (project / loop.CONFIG_PATH).write_text(self.config_text().replace("rounds = 2", 'rounds = "two"'))
        git(project, "commit", "-q", "-am", "break the config on a branch")
        git(project, "checkout", "-q", "main")
        with self.assertRaises(config.ConfigError) as caught:
            loop.run(project, rounds=0, from_ref="broken")
        self.assertIn("on broken: budget.rounds must be a whole number", str(caught.exception))

    def test_failing_check_refuses_to_start(self):
        project = self.make_toy(check="exit 1")
        with self.assertRaises(loop.RunError) as caught:
            loop.run(project, rounds=0)
        self.assertIn("the starting harness does not pass check", str(caught.exception))
        self.assertEqual(git(project, "branch", "--list", "rsi/*"), "")
        self.assertIsNone(loop.find_unfinished(Layout("toy")))

    def test_broken_evaluator_stops_before_optimizing(self):
        project = self.make_toy(run_task="exit 7")
        with self.assertRaises(loop.RunError) as caught:
            loop.run(project, rounds=0)
        self.assertIn("100% of the held-out rollouts failed to run", str(caught.exception))
        stopped = loop.find_unfinished(Layout("toy"))
        self.assertEqual((stopped.status, stopped.step), ("error", "heldout_start"))
        rl = Layout("toy").run(stopped.run)
        self.assertFalse((rl.heldout / stopped.eval_sha).exists())

    def test_new_abandons_an_unfinished_run(self):
        project = self.make_toy(run_task="exit 7")
        with self.assertRaises(loop.RunError):
            loop.run(project, rounds=0)
        layout = Layout("toy")
        first = loop.find_unfinished(layout).run
        (project / loop.CONFIG_PATH).write_text(self.config_text())
        git(project, "commit", "-q", "-am", "fix run_task")
        with self.assertRaises(loop.RunError):
            loop.run(project, rounds=0)  # resumes the old run, which kept its config
        st = loop.run(project, rounds=0, new=True)
        self.assertNotEqual(st.run, first)
        self.assertEqual(st.status, "done")
        self.assertEqual(RunState.load(layout.run(first).state_file).status, "abandoned")
        self.assertFalse((layout.run(first).worktrees / "main").exists())

    def test_a_second_run_is_refused_while_one_is_running(self):
        project = self.make_toy()
        lock = Layout("toy").lock_file
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(str(os.getppid()))
        with self.assertRaises(LockError):
            loop.run(project, rounds=0)

    def test_run_ids_do_not_collide(self):
        project = self.make_toy()
        now = datetime(2026, 11, 20, 9, 0)
        first = loop.run(project, rounds=0, now=now)
        second = loop.run(project, rounds=0, now=now)
        self.assertEqual((first.run, second.run), ("20261120-0900", "20261120-0900-2"))
        self.assertEqual(loop.latest_run(Layout("toy")).run, "20261120-0900-2")

    def test_from_ref_starts_from_another_branch(self):
        project = self.make_toy()
        git(project, "checkout", "-q", "-b", "experiment")
        (project / "rules.txt").write_text("invoice => billing\ncrash => bug\ncharged => billing\n")
        git(project, "commit", "-q", "-am", "one more rule")
        git(project, "checkout", "-q", "main")
        st = loop.run(project, rounds=0, from_ref="experiment")
        self.assertEqual((st.base_ref, st.scores), ("experiment", [50.0]))
