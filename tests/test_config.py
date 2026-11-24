import json
import tempfile
import unittest
from pathlib import Path

from rsi import config

MINIMAL = """
[project]
name = "toy"

[setting]
benchmark = "toy"
model = "model-a"

[harness]
editable = ["rules.txt"]

[commands]
run_task = "python3 score.py {task_file} {checkout} {out}"
"""


class ParseTest(unittest.TestCase):
    def test_minimal_config_gets_defaults(self):
        cfg = config.parse(MINIMAL)
        self.assertEqual(cfg.name, "toy")
        self.assertEqual(cfg.base_branch, "main")
        self.assertEqual(cfg.editable, ("rules.txt",))
        self.assertEqual(cfg.frozen, ())
        self.assertIsNone(cfg.tasks_file)
        self.assertEqual(cfg.heldout_fraction, 0.5)
        self.assertEqual(cfg.rollouts, 1)
        self.assertEqual(cfg.pass_threshold, 0.5)
        self.assertEqual(cfg.heldout, "endpoints")
        self.assertIsNone(cfg.check)
        self.assertIsNone(cfg.ingest)
        self.assertEqual((cfg.rounds, cfg.candidates, cfg.parallel), (5, 3, 2))
        self.assertEqual(cfg.tolerance, 1.0)
        self.assertEqual(cfg.max_infra_failure_rate, 0.3)
        self.assertIsNone(cfg.proposer)
        self.assertTrue(cfg.pull_request)

    def test_all_sections(self):
        cfg = config.parse(MINIMAL + """
[tasks]
file = "~/bench/tasks.jsonl"
heldout_fraction = 0.25
rollouts = 3
pass_threshold = 1
heldout = "every_round"

[budget]
rounds = 2
candidates = 4
parallel = 8
task_timeout_s = 60
agent_timeout_s = 90
max_wall_clock_h = 1.5
tolerance = 0
max_infra_failure_rate = 0.1

[agents]
proposer = "agent {prompt_file}"
selector = "agent {prompt_file}"

[promotion]
pull_request = false
""")
        self.assertEqual(cfg.tasks_file, "~/bench/tasks.jsonl")
        self.assertEqual(cfg.heldout_fraction, 0.25)
        self.assertEqual(cfg.pass_threshold, 1.0)
        self.assertEqual(cfg.heldout, "every_round")
        self.assertEqual(cfg.max_wall_clock_h, 1.5)
        self.assertEqual(cfg.tolerance, 0.0)
        self.assertEqual(cfg.proposer, "agent {prompt_file}")
        self.assertFalse(cfg.pull_request)

    def assert_error(self, text, fragment):
        with self.assertRaises(config.ConfigError) as caught:
            config.parse(text)
        self.assertIn(fragment, str(caught.exception))

    def test_missing_required_key(self):
        self.assert_error(MINIMAL.replace('model = "model-a"', ""), "setting.model is required")

    def test_unknown_key_is_rejected(self):
        self.assert_error(MINIMAL + "\n[budget]\nround = 3\n", "unknown key budget.round")

    def test_unknown_section_is_rejected(self):
        self.assert_error(MINIMAL + "\n[extras]\nx = 1\n", "unknown section [extras]")

    def test_wrong_type(self):
        self.assert_error(MINIMAL + "\n[budget]\nrounds = \"3\"\n", "budget.rounds must be a whole number")

    def test_boolean_is_not_a_number(self):
        self.assert_error(MINIMAL + "\n[budget]\nrounds = true\n", "budget.rounds must be a whole number")

    def test_editable_must_be_a_list_of_strings(self):
        self.assert_error(MINIMAL.replace('["rules.txt"]', '"rules.txt"'), "harness.editable must be a list of strings")

    def test_ranges(self):
        self.assert_error(MINIMAL + "\n[tasks]\nheldout_fraction = 1\n", "tasks.heldout_fraction")
        self.assert_error(MINIMAL + "\n[tasks]\nrollouts = 0\n", "tasks.rollouts")
        self.assert_error(MINIMAL + "\n[tasks]\nheldout = \"sometimes\"\n", "tasks.heldout")
        self.assert_error(MINIMAL + "\n[budget]\ncandidates = 0\n", "budget.candidates")
        self.assert_error(MINIMAL.replace('editable = ["rules.txt"]', "editable = []"), "harness.editable")
        self.assert_error(MINIMAL.replace('name = "toy"', 'name = "my toy"'), "project.name")
        for unsafe in ("..", ".", ".hidden", "-x"):
            self.assert_error(MINIMAL.replace('name = "toy"', f'name = "{unsafe}"'), "project.name")
        self.assert_error(MINIMAL.replace('benchmark = "toy"', 'benchmark = ".."'), "setting.benchmark")

    def test_task_file_path_must_not_depend_on_the_current_directory(self):
        self.assert_error(MINIMAL + '\n[tasks]\nfile = "bench/tasks.jsonl"\n', "tasks.file must be an absolute path")
        self.assert_error(MINIMAL + '\n[tasks]\nfile = "../tasks.jsonl"\n', "tasks.file must be an absolute path")
        self.assertEqual(config.parse(MINIMAL + '\n[tasks]\nfile = "/bench/t.jsonl"\n').tasks_file, "/bench/t.jsonl")

    def test_invalid_toml(self):
        self.assert_error("[project\nname=", "not valid TOML")

    def test_new_keys_default_off(self):
        cfg = config.parse(MINIMAL)
        self.assertEqual((cfg.repo, cfg.paired, cfg.min_gain, cfg.constraints), (None, False, None, ()))
        self.assertFalse(cfg.confirm)
        self.assertEqual((cfg.agent_cli, cfg.agent_model, cfg.agent_effort), ("claude", None, None))
        self.assertEqual((cfg.promotion_mode, cfg.draft, cfg.branch_prefix), ("per_run", False, "rsi/"))

    MEASURED = MINIMAL.replace('name = "toy"', 'name = "toy"\nrepo = "~/work/toy"') + """
[tasks]
paired = true
[budget]
min_gain = 1
[[constraints]]
metric = "cost_usd"
max_increase_pct = 10
[[constraints]]
metric = "accuracy_pct"
max_decrease_pct = 0
[agents]
cli = "agy"
model = "gemini-4-neon"
effort = "high"
[promotion]
mode = "per_edit"
draft = true
branch_prefix = "sk-rsi-"
"""

    def test_measured_goal_config(self):
        cfg = config.parse(self.MEASURED)
        self.assertEqual(cfg.constraints, (config.Constraint("cost_usd", "increase", 10.0),
                                           config.Constraint("accuracy_pct", "decrease", 0.0)))
        self.assertEqual((cfg.paired, cfg.min_gain, cfg.repo), (True, 1.0, "~/work/toy"))
        self.assertEqual((cfg.agent_cli, cfg.agent_model, cfg.agent_effort), ("agy", "gemini-4-neon", "high"))
        self.assertEqual((cfg.promotion_mode, cfg.draft, cfg.branch_prefix), ("per_edit", True, "sk-rsi-"))
        self.assertEqual(config.Config.from_dict(json.loads(json.dumps(cfg.to_dict()))), cfg)

    def test_from_dict_fills_new_keys(self):
        old = config.parse(MINIMAL).to_dict()
        for key in ("repo", "paired", "min_gain", "constraints", "agent_cli", "agent_model", "agent_effort",
                    "promotion_mode", "draft", "branch_prefix", "confirm"):
            del old[key]
        self.assertEqual(config.Config.from_dict(old), config.parse(MINIMAL))

    def test_new_key_errors(self):
        entry = lambda body: MINIMAL + "\n[[constraints]]\n" + body + "\n"
        external = MINIMAL.replace('name = "toy"', 'name = "toy"\nrepo = "~/t"')
        cases = [
            (entry("max_increase_pct = 10"), "constraints[0].metric is required"),
            (entry('metric = "Cost"\nmax_increase_pct = 10'), "constraints[0].metric"),
            (entry('metric = "cost"\nmax_increase_pct = 1\nmax_decrease_pct = 1'),
             "exactly one of max_increase_pct or max_decrease_pct"),
            (entry('metric = "cost"'), "exactly one of max_increase_pct or max_decrease_pct"),
            (entry('metric = "cost"\nmax_increase_pct = -1'), "must be a number, 0 or more"),
            (entry('metric = "cost"\nmax_increase_pct = "10"'), "must be a number, 0 or more"),
            (entry('metric = "cost"\nmax_increase_pct = true'), "must be a number, 0 or more"),
            (entry('metric = "cost"\nmax_increase_pct = 1\nlimit = 2'), "unknown key constraints[0].limit"),
            (entry('metric = "cost"\nmax_increase_pct = 1') + '[[constraints]]\nmetric = "cost"\nmax_increase_pct = 2\n',
             "constraints[1] repeats"),
            ('constraints = "x"\n' + MINIMAL, "constraints must be a list of tables"),
            (MINIMAL + '\n[tasks]\npaired = true\nheldout = "every_round"\n',
             'tasks.paired cannot be combined with tasks.heldout = "every_round"'),
            (MINIMAL + "\n[budget]\nmin_gain = -1\n", "budget.min_gain"),
            (MINIMAL + "\n[tasks]\nconfirm = true\n", "tasks.confirm needs tasks.paired"),
            (MINIMAL + '\n[agents]\ncli = "codex"\n', 'agents.cli must be "claude" or "agy"'),
            (MINIMAL + '\n[agents]\ncli = "agy"\n', 'agents.model is required when agents.cli is "agy"'),
            (MINIMAL + '\n[promotion]\nmode = "per_commit"\n', 'promotion.mode must be "per_run" or "per_edit"'),
            (MINIMAL + '\n[promotion]\nmode = "per_edit"\n',
             'promotion.mode = "per_edit" needs a config kept outside the repository'),
            (external + '\n[promotion]\nbranch_prefix = "/x"\n', "promotion.branch_prefix"),
            (external + '\n[promotion]\nbranch_prefix = "a b"\n', "promotion.branch_prefix"),
        ]
        for text, fragment in cases:
            with self.subTest(fragment=fragment):
                self.assert_error(text, fragment)

    def test_round_trip_through_dict(self):
        cfg = config.parse(MINIMAL)
        self.assertEqual(config.Config.from_dict(cfg.to_dict()), cfg)


class LoadTest(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(config.ConfigError) as caught:
            config.load(Path("/nonexistent/.rsi/config.toml"))
        self.assertIn("not found", str(caught.exception))

    def test_reads_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(MINIMAL)
            self.assertEqual(config.load(path).name, "toy")
