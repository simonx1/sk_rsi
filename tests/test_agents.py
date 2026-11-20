import json
import tempfile
import unittest
from pathlib import Path

from rsi import agents


class RunAgentTest(unittest.TestCase):
    def setUp(self):
        # Resolved, so a path here has one name; the symlink test builds its own second name.
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cwd = self.tmp / "work tree"
        self.cwd.mkdir()
        self.out = self.tmp / "agent out"
        self.kwargs = dict(
            cwd=self.cwd, out_dir=self.out, feedback_dir=self.tmp / "feedback",
            history_dir=self.tmp / "history", deny=[self.tmp / "private"], timeout=20,
        )

    def test_command_receives_prompt_paths_and_environment(self):
        command = (
            "cat {prompt_file} > {out_dir}/seen.txt; "
            "echo $RSI_ROLE $RSI_ROUND $RSI_SAMPLE > {out_dir}/env.txt; "
            "pwd > {out_dir}/cwd.txt; "
            "ls -d {feedback_dir} {history_dir} {settings_file} > /dev/null; "
            "echo talking"
        )
        result = agents.run_agent(
            "proposer", command, "the prompt", env={"RSI_ROUND": "2", "RSI_SAMPLE": "s1"}, **self.kwargs)
        self.assertEqual(result, agents.AgentResult(True, None))
        self.assertEqual((self.out / "seen.txt").read_text(), "the prompt")
        self.assertEqual((self.out / "env.txt").read_text().strip(), "proposer 2 s1")
        self.assertEqual(Path((self.out / "cwd.txt").read_text().strip()).resolve(), self.cwd.resolve())
        self.assertIn("talking", (self.out / "agent.log").read_text())

    def test_settings_deny_reading_and_editing_every_given_path(self):
        task_file = self.tmp / "bench" / "tasks.jsonl"
        task_file.parent.mkdir()
        task_file.write_text("")
        kwargs = dict(self.kwargs, deny=[self.tmp / "private", task_file])
        agents.run_agent("selector", "true", "p", **kwargs)
        settings = json.loads((self.out / "settings.json").read_text())
        private = str((self.tmp / "private").resolve()).lstrip("/")
        single = str(task_file.resolve()).lstrip("/")
        # claude applies Read and Edit deny rules to files; it ignores Write rules and warns about them.
        self.assertEqual(settings, {"permissions": {"deny": [
            f"Read(//{private}/**)", f"Edit(//{private}/**)",
            f"Read(//{single})", f"Edit(//{single})",
        ]}})

    def test_a_symlinked_path_is_denied_under_both_names(self):
        real = self.tmp / "real"
        (real / "private").mkdir(parents=True)
        link = self.tmp / "link"
        link.symlink_to(real)
        agents.run_agent("selector", "true", "p", **dict(self.kwargs, deny=[link / "private"]))
        deny = json.loads((self.out / "settings.json").read_text())["permissions"]["deny"]
        as_given = str(link.absolute() / "private").lstrip("/")
        resolved = str((real / "private").resolve()).lstrip("/")
        self.assertNotEqual(as_given, resolved)
        self.assertEqual(deny, [
            f"Read(//{as_given}/**)", f"Edit(//{as_given}/**)",
            f"Read(//{resolved}/**)", f"Edit(//{resolved}/**)",
        ])

    def test_failure_and_timeout(self):
        self.assertEqual(agents.run_agent("proposer", "exit 4", "p", **self.kwargs).error, "exit code 4")
        slow = dict(self.kwargs, timeout=1)
        self.assertEqual(agents.run_agent("proposer", "sleep 30", "p", **slow).error, "timed out after 1s")

    def test_default_command_names_every_placeholder(self):
        for name in ("prompt_file", "out_dir", "feedback_dir", "history_dir", "settings_file"):
            self.assertIn("{" + name + "}", agents.DEFAULT_COMMAND)
        self.assertIn("--disallowedTools Bash", agents.DEFAULT_COMMAND)

    def test_default_command_confines_the_agent(self):
        # --restricted confines the file tools to the working directory and the added directories and
        # ignores user settings; --tools fixes which tools exist; no MCP servers are loaded.
        for flag in ("--restricted", "--strict-mcp-config", "--tools Read,Edit,Write,Grep,Glob"):
            self.assertIn(flag, agents.DEFAULT_COMMAND)

    def test_read_only_paths_may_be_read_but_not_edited(self):
        feedback = self.tmp / "feedback"
        kwargs = dict(self.kwargs, readonly=[feedback])
        agents.run_agent("selector", "true", "p", **kwargs)
        deny = json.loads((self.out / "settings.json").read_text())["permissions"]["deny"]
        private = str(self.tmp / "private").lstrip("/")
        self.assertEqual(deny, [
            f"Read(//{private}/**)", f"Edit(//{private}/**)",
            "Edit(//" + str(feedback).lstrip("/") + "/**)",
        ])


class ManifestTest(unittest.TestCase):
    GOOD = {"root_cause": {"new": " premature completion "}, "what": " add a gate ",
            "predicted": ["t1", "t2", "t1"], "risk": "may block finished work"}

    def test_normalised(self):
        self.assertEqual(agents.validate_manifest(self.GOOD), {
            "root_cause": {"new": "premature completion"}, "what": "add a gate",
            "predicted": ["t1", "t2"], "risk": "may block finished work",
        })

    def test_optional_fields_default(self):
        got = agents.validate_manifest({"root_cause": {"id": "P3"}, "what": "x"})
        self.assertEqual((got["predicted"], got["risk"]), ([], ""))

    def test_rejections(self):
        bad = [
            ([1], "must be a JSON object"),
            ({"what": "x"}, "root_cause"),
            ({"root_cause": {"id": "P1", "new": "x"}, "what": "x"}, "root_cause"),
            ({"root_cause": {"other": "x"}, "what": "x"}, "root_cause"),
            ({"root_cause": {"id": ""}, "what": "x"}, "root_cause"),
            ({"root_cause": {"id": "P1"}}, "what"),
            ({"root_cause": {"id": "P1"}, "what": "  "}, "what"),
            ({"root_cause": {"id": "P1"}, "what": "x", "predicted": "t1"}, "predicted"),
            ({"root_cause": {"id": "P1"}, "what": "x", "predicted": [1]}, "predicted"),
            ({"root_cause": {"id": "P1"}, "what": "x", "risk": 3}, "risk"),
        ]
        for obj, fragment in bad:
            with self.assertRaises(agents.AgentOutputError, msg=str(obj)) as caught:
                agents.validate_manifest(obj)
            self.assertIn(fragment, str(caught.exception))


class OutputFilesTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, name, text):
        (self.out / name).write_text(text)

    def assert_error(self, call, fragment):
        with self.assertRaises(agents.AgentOutputError) as caught:
            call()
        self.assertIn(fragment, str(caught.exception))

    def test_manifest_file(self):
        self.assert_error(lambda: agents.read_manifest(self.out), "manifest.json was not written")
        self.write("manifest.json", "not json")
        self.assert_error(lambda: agents.read_manifest(self.out), "manifest.json is not valid JSON")
        self.write("manifest.json", json.dumps({"root_cause": {"id": "P1"}, "what": "x"}))
        self.assertEqual(agents.read_manifest(self.out)["root_cause"], {"id": "P1"})

    def test_screen_file(self):
        self.write("screen.json", json.dumps({"screen": [
            {"sample": "s1", "keep": False, "reason": "repeats a failed attempt"},
            {"sample": "s9", "keep": False},
        ]}))
        self.assertEqual(agents.read_screen(self.out, ["s1", "s2"]), {
            "s1": (False, "repeats a failed attempt"),
            "s2": (True, "not mentioned by the selector"),
        })
        self.write("screen.json", json.dumps({"screen": [{"sample": [], "keep": False}, "junk"]}))
        self.assertEqual(agents.read_screen(self.out, ["s1"]), {"s1": (True, "not mentioned by the selector")})
        self.write("screen.json", json.dumps({"screen": [{"sample": "s1", "keep": "no"}]}))
        self.assert_error(lambda: agents.read_screen(self.out, ["s1"]), "keep")
        self.write("screen.json", json.dumps({"screen": "all"}))
        self.assert_error(lambda: agents.read_screen(self.out, ["s1"]), "screen")

    def test_decision_file(self):
        self.write("decision.json", json.dumps({"accept": "s2", "reason": "best score"}))
        self.assertEqual(agents.read_decision(self.out, ["s1", "s2"]), ("s2", "best score"))
        self.write("decision.json", json.dumps({"accept": None}))
        self.assertEqual(agents.read_decision(self.out, ["s1"]), (None, ""))
        self.write("decision.json", json.dumps({"accept": "s7"}))
        self.assert_error(lambda: agents.read_decision(self.out, ["s1"]), "s7")
        self.write("decision.json", json.dumps({"reason": "forgot"}))
        self.assert_error(lambda: agents.read_decision(self.out, ["s1"]), "accept")
