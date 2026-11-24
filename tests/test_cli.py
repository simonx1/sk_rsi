import contextlib
import io
import json
import shlex

from rsi import cli, config, loop, sim
from rsi.state import Layout, RunState
from tests.helpers import git, make_repo
from tests.toy import ToyCase


class CliCase(ToyCase):
    def rsi(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()


class InitTest(CliCase):
    def test_writes_a_config_that_loads(self):
        repo = make_repo(self.tmp / "My Project", {"README.md": "x\n"})
        code, out, _ = self.rsi("init", "--project", repo)
        self.assertEqual(code, 0)
        cfg = config.load(repo / ".rsi" / "config.toml")
        self.assertEqual((cfg.name, cfg.benchmark, cfg.base_branch), ("My-Project", "My-Project", "main"))
        self.assertTrue((self.home / "My-Project" / "private").is_dir())
        self.assertIn("tasks.jsonl", out)

    def test_init_external(self):
        repo = make_repo(self.tmp / "My Project", {"README.md": "x\n"})
        code, out, _ = self.rsi("init", "--external", "--project", repo)
        self.assertEqual(code, 0)
        path = self.home / "My-Project" / "config.toml"
        self.assertIn(str(path), out)
        self.assertFalse((repo / ".rsi").exists())
        cfg = config.load(path)
        self.assertEqual((cfg.name, cfg.repo), ("My-Project", str(repo)))
        self.assertTrue((self.home / "My-Project" / "private").is_dir())

    def test_init_external_with_a_name(self):
        repo = make_repo(self.tmp / "repo", {"README.md": "x\n"})
        self.assertEqual(self.rsi("init", "--external", "--name", "latency", "--project", repo)[0], 0)
        self.assertEqual(config.load(self.home / "latency" / "config.toml").name, "latency")

    def test_init_external_refuses_when_a_config_exists(self):
        repo = make_repo(self.tmp / "repo", {"README.md": "x\n"})
        self.rsi("init", "--external", "--project", repo)
        code, _, err = self.rsi("init", "--external", "--project", repo)
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        code, _, err = self.rsi("init", "--project", repo)
        self.assertEqual(code, 2)
        self.assertIn("already has a config", err)

    def test_refuses_to_overwrite(self):
        repo = make_repo(self.tmp / "repo", {"README.md": "x\n"})
        self.rsi("init", "--project", repo)
        code, _, err = self.rsi("init", "--project", repo)
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)


class ValidateTest(CliCase):
    def test_valid_project(self):
        project = self.make_toy()
        code, out, _ = self.rsi("validate", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("evolve tasks: 6, held-out tasks: 6", out)
        self.assertIn("check: passed", out)
        self.assertIn("warning: a custom agent command", out)

    def test_validate_does_not_fix_the_split(self):
        project = self.make_toy()
        split_file = Layout("toy").split_file
        split_file.unlink()
        code, out, _ = self.rsi("validate", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("evolve tasks: 6, held-out tasks: 6", out)
        self.rsi("run", "--dry-run", "--project", project)
        self.assertFalse(split_file.exists())  # only a real run assigns sides for good

    def test_invalid_config_exits_2(self):
        project = self.make_toy()
        (project / ".rsi" / "config.toml").write_text("[project\n")
        code, _, err = self.rsi("validate", "--project", project)
        self.assertEqual(code, 2)
        self.assertIn("not valid TOML", err)

    def test_validation_errors_exit_2(self):
        project = self.make_toy()
        Layout("toy").tasks_path(None).unlink()
        code, out, _ = self.rsi("validate", "--project", project)
        self.assertEqual(code, 2)
        self.assertIn("error: task file not found", out)

    def test_failing_check_exits_2(self):
        project = self.make_toy(check="exit 1")
        code, out, _ = self.rsi("validate", "--project", project)
        self.assertEqual(code, 2)
        self.assertIn("check failed with exit code 1", out)


class RunTest(CliCase):
    def test_dry_run_prints_the_cost_and_starts_nothing(self):
        project = self.make_toy()
        code, out, _ = self.rsi("run", "--dry-run", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("up to 54 task runs", out)
        self.assertIn("6 evolve tasks, 6 held-out tasks, 1 rollout", out)
        self.assertIsNone(loop.latest_run(Layout("toy")))
        _, out, _ = self.rsi("run", "--dry-run", "--rounds", "0", "--project", project)
        self.assertIn("up to 18 task runs", out)

    def test_run_then_status(self):
        project = self.make_toy()
        code, out, err = self.rsi("run", "--project", project)
        self.assertEqual(code, 0)
        for text in (out, self.rsi("status", "--project", project)[1]):
            self.assertIn("Evolve: 33.33 → 66.67 → 83.33", text)
            self.assertIn("Held-out: 33.33 → 83.33", text)
            self.assertIn("Setting: toy-v1, model-a", text)
            self.assertIn("Round 2 of 2", text)
            self.assertIn("Pull request: not opened: the project has no git remote", text)
            self.assertIn("  git push -u origin rsi/", text)
            self.assertIn("Pending tasks from live outcomes: 0", text)
        self.assertIn("[rsi] round 1: accepted s1", err)

    def test_status_shows_heldout_after_every_round_when_it_was_scored(self):
        project = self.make_toy(heldout="every_round")
        self.rsi("run", "--no-promote", "--project", project)
        _, out, _ = self.rsi("status", "--project", project)
        self.assertIn("Held-out: 33.33 → 66.67 → 83.33", out)

    def test_malformed_config_exits_2_from_run(self):
        project = self.make_toy()
        (project / ".rsi" / "config.toml").write_text("[project\n")
        code, _, err = self.rsi("run", "--project", project)
        self.assertEqual(code, 2)
        self.assertIn("not valid TOML", err)

    def test_commands_work_for_an_external_project(self):
        project = self.make_toy(external=True)
        self.assertEqual(self.rsi("validate", "--project", project)[0], 0)
        code, out, _ = self.rsi("run", "--no-promote", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("Evolve: 33.33 → 66.67 → 83.33", out)
        self.assertIn("Evolve: 33.33 → 66.67 → 83.33", self.rsi("status", "--project", project)[1])
        _, out, _ = self.rsi("sim", "show", "--project", project)
        self.assertIn("P1  rules cover too few keywords per label", out)
        code, out, _ = self.rsi("promote", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("the project has no git remote", out)

    def test_status_without_runs(self):
        project = self.make_toy()
        self.assertEqual(self.rsi("status", "--project", project)[:2], (0, "No runs yet.\n"))

    def test_a_run_that_stops_on_an_error_exits_1(self):
        project = self.make_toy(run_task="exit 7")
        code, _, err = self.rsi("run", "--project", project)
        self.assertEqual(code, 1)
        self.assertIn("rollouts failed to run", err)
        _, out, _ = self.rsi("status", "--project", project)
        self.assertIn(": error (", out)

    def test_no_promote_then_promote(self):
        project = self.make_toy()
        _, out, _ = self.rsi("run", "--no-promote", "--project", project)
        self.assertIn("Pull request: not opened: --no-promote", out)
        code, out, _ = self.rsi("promote", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("Pull request: not opened: the project has no git remote", out)
        self.assertEqual(loop.latest_run(Layout("toy")).promoted["skipped"], "the project has no git remote")

    def test_promote_again_keeps_the_pull_request_already_opened(self):
        project = self.make_toy()
        self.rsi("run", "--no-promote", "--project", project)
        layout = Layout("toy")
        st = loop.latest_run(layout)
        st.promoted = {"url": "https://example.test/toy/pull/1", "draft": False}
        st.save(layout.run(st.run).state_file)
        code, out, _ = self.rsi("promote", "--project", project)
        self.assertEqual(code, 0)
        self.assertIn("Pull request: https://example.test/toy/pull/1", out)
        self.assertEqual(loop.latest_run(layout).promoted["url"], "https://example.test/toy/pull/1")

    def test_negative_rounds_are_refused(self):
        project = self.make_toy()
        for command in ("run", "continue"):
            with self.assertRaises(SystemExit) as caught:
                self.rsi(command, "--dry-run", "--rounds", "-5", "--project", project)
            self.assertEqual(caught.exception.code, 2)

    def test_promote_an_unknown_run(self):
        project = self.make_toy()
        code, _, err = self.rsi("promote", "nope", "--project", project)
        self.assertEqual(code, 1)
        self.assertIn("no run named nope", err)

    def test_outside_a_git_repo(self):
        code, _, err = self.rsi("status", "--project", self.tmp)
        self.assertEqual(code, 1)
        self.assertIn("rsi:", err)

    def test_bad_arguments_exit_2(self):
        with self.assertRaises(SystemExit) as caught:
            self.rsi("frobnicate")
        self.assertEqual(caught.exception.code, 2)


class LiveOutcomesTest(CliCase):
    def setUp(self):
        super().setUp()
        feed = self.tmp / "feed.jsonl"
        feed.write_text("".join(json.dumps(r) + "\n" for r in [
            {"kind": "evidence", "id": "e1", "summary": "Customers keep writing charged", "task_id": "t03"},
            {"kind": "task", "id": "live-1", "summary": "worth replaying",
             "task": {"id": "t13", "text": "bill me twice", "label": "billing"}},
        ]) + "noise\n")
        self.project = self.make_toy(ingest=f"cat {shlex.quote(str(feed))}")

    def test_ingest(self):
        code, out, _ = self.rsi("ingest", "--project", self.project)
        self.assertEqual(code, 0)
        self.assertIn("1 evidence record, 1 new task queued, 1 line skipped", out)
        self.assertIn("line 3: not JSON", out)

    def test_continue_folds_pending_tasks_into_a_new_benchmark_version(self):
        self.rsi("ingest", "--project", self.project)
        code, out, _ = self.rsi("continue", "--rounds", "0", "--project", self.project)
        self.assertEqual(code, 0)
        self.assertIn("added 1 task; the benchmark is now toy-v2", out)
        st = loop.latest_run(Layout("toy"))
        self.assertEqual(st.benchmark, "toy-v2")
        self.assertEqual(len(st.evolve_ids) + len(st.heldout_ids), 13)
        self.assertIn("Pending tasks from live outcomes: 0", out)

    def test_continue_refuses_while_a_run_is_unfinished(self):
        (self.project / ".rsi" / "config.toml").write_text(self.config_text(run_task="exit 7"))
        git(self.project, "commit", "-q", "-am", "break run_task")
        self.rsi("run", "--project", self.project)
        code, _, err = self.rsi("continue", "--project", self.project)
        self.assertEqual(code, 1)
        self.assertIn("is unfinished", err)


class SimCommandsTest(CliCase):
    def setUp(self):
        super().setUp()
        self.project = self.make_toy()
        self.path = self.project / ".rsi" / "sim.json"

    def seed(self):
        memory = sim.empty()
        for round_no, name in enumerate(("stops too early", "stops before verifying")):
            sim.record_attempt(memory, {"new": name}, {
                "run": "r1", "round": round_no, "what": "edit", "predicted": [], "fixed": [], "missed": [],
                "regressed": [], "passed": 1, "total": 2, "score_before": 0.0, "score_after": 50.0,
                "domain": {"benchmark": "toy-v1", "model": "model-a"}})
        sim.save(self.path, memory)

    def test_show(self):
        self.assertEqual(self.rsi("sim", "show", "--project", self.project)[:2], (0, "The memory is empty.\n"))
        self.seed()
        _, out, _ = self.rsi("sim", "show", "--project", self.project)
        self.assertIn("P1  stops too early", out)
        self.assertIn("P2  stops before verifying", out)

    def test_merge(self):
        self.seed()
        code, out, _ = self.rsi("sim", "merge", "P2", "P1", "--project", self.project)
        self.assertEqual(code, 0)
        self.assertIn("merged P2 into P1", out)
        memory = sim.load(self.path)
        self.assertEqual(memory["aliases"], {"P2": "P1"})
        self.assertEqual(len(memory["problems"][0]["attempts"]), 2)

    def test_merge_of_an_unknown_problem_exits_1(self):
        self.seed()
        code, _, err = self.rsi("sim", "merge", "P9", "P1", "--project", self.project)
        self.assertEqual(code, 1)
        self.assertIn("unknown problem: P9", err)
