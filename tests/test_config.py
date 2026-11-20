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
