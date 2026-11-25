import os
import re
import subprocess
import unittest
from pathlib import Path

from rsi import cli, config

ROOT = Path(__file__).resolve().parents[1]
MINIMAL_FOR_DOCS = """[project]
name = "docs"
[setting]
benchmark = "docs"
model = "m"
[harness]
editable = ["src/**"]
[commands]
run_task = "true"
"""


def documented_commands(text):
    """The first word after `rsi` in each code line of a document."""
    found = set()
    for line in text.splitlines():
        match = re.match(r"\s*(?:\$ )?rsi ([a-z]+)", line)
        if match:
            found.add(match.group(1))
    return found


class DocsTest(unittest.TestCase):
    def setUp(self):
        parser = cli.build_parser()
        self.commands = set(parser._subparsers._group_actions[0].choices)

    def test_readme_and_skill_only_name_real_commands(self):
        for name in ("README.md", "skill/SKILL.md"):
            text = (ROOT / name).read_text()
            used = documented_commands(text)
            self.assertTrue(used, f"{name} shows no rsi commands")
            self.assertEqual(used - self.commands, set(), name)

    def test_readme_covers_every_command(self):
        used = documented_commands((ROOT / "README.md").read_text())
        self.assertEqual(self.commands - used, set())

    def test_readme_config_example_loads(self):
        text = (ROOT / "README.md").read_text()
        block = re.search(r"```toml\n(.*?)```", text, re.S).group(1)
        cfg = config.parse(block)
        self.assertEqual(cfg.name, "ticket-triage")

    NEW_KEYS = ("repo =", "paired =", "min_gain =", "[[constraints]]", "max_increase_pct", "max_decrease_pct",
                "cli =", "mode =", "draft =", "branch_prefix =", "confirm =")

    def measured_section(self):
        text = (ROOT / "README.md").read_text()
        start = text.index("## Measured goals")
        end = text.index("\n## ", start + 1)
        return text[start:end]

    def test_readme_measured_goals_example_loads(self):
        section = self.measured_section()
        cfg = config.parse(re.search(r"```toml\n(.*?)```", section, re.S).group(1))
        self.assertTrue(cfg.paired)
        self.assertEqual([(c.metric, c.direction) for c in cfg.constraints],
                         [("cost_usd", "increase"), ("accuracy_pct", "decrease")])
        self.assertEqual((cfg.agent_cli, cfg.promotion_mode), ("agy", "per_edit"))
        self.assertIsNotNone(cfg.repo)
        self.assertIsNotNone(cfg.min_gain)

    def test_readme_names_every_new_key(self):
        section = self.measured_section()
        for key in self.NEW_KEYS + ("rsi init --external", '"metrics"', '"expected"', '"title"', "RSI_CURRENT"):
            self.assertIn(key, section, key)

    def test_template_mentions_every_new_key_and_still_loads(self):
        for key in self.NEW_KEYS:
            self.assertIn(key, cli.TEMPLATE, key)
        cfg = config.parse(cli.TEMPLATE.replace("{name}", "example"))
        self.assertEqual((cfg.paired, cfg.constraints, cfg.agent_cli, cfg.promotion_mode), (False, (), "claude", "per_run"))

    def test_nothing_about_an_application_of_the_framework_is_in_the_repository(self):
        """The repository is public: it holds the framework, not what it is used on.

        The words to look for would themselves name those applications, so they live outside the
        repository: one per line in the file $RSI_BANNED_WORDS names, ~/.rsi/banned-words by default.
        Lines starting with # are comments. Without that file the test is skipped."""
        words_file = Path(os.environ.get("RSI_BANNED_WORDS", "~/.rsi/banned-words")).expanduser()
        if not words_file.is_file():
            self.skipTest(f"no {words_file}")
        if not (ROOT / ".git").exists():
            self.skipTest("not a git checkout")
        words = sorted({line.strip().lower() for line in words_file.read_text().splitlines()
                        if line.strip() and not line.strip().startswith("#")})
        git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        texts = {name: (ROOT / name).read_text(errors="replace")
                 for name in git("ls-files", "-z").split("\0") if name and (ROOT / name).is_file()}
        texts["commit messages"] = git("log", "--format=%B", "HEAD")
        found = [f"{word} in {where}" for where, text in texts.items() for word in words if word in text.lower()]
        self.assertEqual(found, [])

    def test_readme_documents_the_progress_line(self):
        from rsi import progress
        text = (ROOT / "README.md").read_text()
        start = text.index("## Progress")
        section = text[start:text.index("\n## ", start + 1)]
        for name in sorted(progress.FIELDS) + ["metric.<name>", "limit.<name>", '"failures"', '"none"',
                                               "log_interval_s", "interval_s"]:
            self.assertIn(name, section, name)
        cfg = config.parse(re.search(r"```toml\n(.*?)```", section, re.S).group(1)
                           .replace("[progress]", MINIMAL_FOR_DOCS + "\n[progress]", 1))
        self.assertEqual(cfg.progress_events, "all")

    def test_readme_and_template_document_agent_notes(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("**Notes for the agents.**", text)
        self.assertIn('notes = "agent-notes.md"', text)
        self.assertIn("# notes = ", cli.TEMPLATE)

    def test_readme_explains_earlier_attempts(self):
        text = (ROOT / "README.md").read_text()
        how = " ".join(text[text.index("## How it works"):text.index("## Install")].split())
        self.assertIn("Edits that were not kept", how)
        self.assertIn("earlier runs", how)

    def test_readme_documents_the_median(self):
        self.assertIn('**Median of runs.** With `aggregate = "median"`', (ROOT / "README.md").read_text())

    def test_template_mentions_progress(self):
        self.assertIn("# [progress]", cli.TEMPLATE)
        self.assertIn("# line = ", cli.TEMPLATE)

    def test_skill_has_front_matter(self):
        text = (ROOT / "skill" / "SKILL.md").read_text()
        self.assertTrue(text.startswith("---\nname: rsi\ndescription: "))

    def test_install_script_is_executable(self):
        script = ROOT / "bin" / "install-skill"
        self.assertTrue(script.stat().st_mode & 0o111)
