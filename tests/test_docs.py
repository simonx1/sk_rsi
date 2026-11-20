import re
import unittest
from pathlib import Path

from rsi import cli, config

ROOT = Path(__file__).resolve().parents[1]


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

    def test_skill_has_front_matter(self):
        text = (ROOT / "skill" / "SKILL.md").read_text()
        self.assertTrue(text.startswith("---\nname: rsi\ndescription: "))

    def test_install_script_is_executable(self):
        script = ROOT / "bin" / "install-skill"
        self.assertTrue(script.stat().st_mode & 0o111)
