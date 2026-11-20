"""Builds the toy project in a temporary git repo, with RSI_HOME pointed at a temporary directory."""

import os
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import git

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"
PYTHON = shlex.quote(sys.executable)

CONFIG = """\
[project]
name = "toy"
base_branch = "main"

[setting]
benchmark = "toy"
model = "{model}"

[harness]
editable = ["rules.txt"]
frozen = ["classify.py", "score.py"]

[tasks]
rollouts = 1
heldout = "{heldout}"

[commands]
run_task = {run_task}
{optional_commands}
[budget]
rounds = {rounds}
candidates = {candidates}
parallel = 2
task_timeout_s = 60
agent_timeout_s = {agent_timeout_s}
{agents}"""


def toml_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class ToyCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.home = self.tmp / "home"
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(self.home)}))

    def use_script(self, name: str) -> None:
        os.environ["FAKE_AGENT_SCRIPT"] = str(TOY / name)

    def config_text(self, **settings) -> str:
        values = {
            "model": "model-a",
            "heldout": "endpoints",
            "rounds": 2,
            "candidates": 3,
            "run_task": f"{PYTHON} score.py {{task_file}} {{checkout}} {{out}}",
            "check": f"{PYTHON} -c \"import ast; ast.parse(open('classify.py').read())\"",
            "ingest": None,
            "agents": True,
            "agent_timeout_s": 120,
        }
        values.update(settings)
        optional = ""
        for name in ("check", "ingest"):
            if values[name]:
                optional += f"{name} = {toml_string(values[name])}\n"
        agents = ""
        if values["agents"]:
            command = toml_string(f"{PYTHON} {shlex.quote(str(TOY / 'fake_agent.py'))} --out {{out_dir}}")
            agents = f"\n[agents]\nproposer = {command}\nselector = {command}\n"
        return CONFIG.format(
            model=values["model"], heldout=values["heldout"], rounds=values["rounds"],
            candidates=values["candidates"], run_task=toml_string(values["run_task"]),
            agent_timeout_s=values["agent_timeout_s"], optional_commands=optional, agents=agents,
        )

    def make_toy(self, script: str = "script.json", **settings) -> Path:
        project = self.tmp / "project"
        shutil.copytree(TOY / "project", project)
        (project / ".rsi").mkdir()
        (project / ".rsi" / "config.toml").write_text(self.config_text(**settings))
        private = self.home / "toy" / "private"
        private.mkdir(parents=True)
        shutil.copy(TOY / "bench" / "tasks.jsonl", private / "tasks.jsonl")
        shutil.copy(TOY / "bench" / "split.json", private / "split.json")
        git(project, "init", "-q", "-b", "main")
        git(project, "config", "user.email", "test@example.com")
        git(project, "config", "user.name", "Test")
        git(project, "config", "commit.gpgsign", "false")
        git(project, "add", "-A")
        git(project, "commit", "-q", "-m", "toy project")
        self.use_script(script)
        return project
