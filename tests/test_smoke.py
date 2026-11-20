"""One round on the toy project with the real default agents. Costs tokens, so it is opt-in."""

import os
import shutil
import unittest

from rsi import loop
from rsi.state import Layout
from tests.toy import ToyCase


@unittest.skipUnless(os.environ.get("RSI_SMOKE") == "1", "set RSI_SMOKE=1 to run with the real claude CLI")
class SmokeTest(ToyCase):
    def test_one_round_with_the_default_agents(self):
        self.assertIsNotNone(shutil.which("claude"), "the claude CLI is not on PATH")
        project = self.make_toy(agents=False, rounds=1, candidates=1, agent_timeout_s=900)
        st = loop.run(project, promote=False)
        self.assertEqual(st.status, "done")
        rl = Layout("toy").run(st.run)
        log = (rl.agents / "round-0" / "s1" / "agent.log").read_text()
        entry = st.rounds_log[0]
        for dropped in entry["dropped"]:
            self.assertNotIn("agent failed", dropped["reason"], log)
            self.assertNotIn("invalid manifest", dropped["reason"], log)
        self.assertTrue((rl.agents / "round-0" / "s1" / "manifest.json").is_file(), log)
        if not entry["dropped"]:
            self.assertTrue((rl.agents / "round-0" / "select" / "decision.json").is_file())
        print(f"\nsmoke: evolve {st.scores[0]} -> {st.scores[-1]}, accepted {entry['accepted']}, "
              f"reason: {entry['reason']}")
