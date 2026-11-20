import json
import os
import shlex
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from rsi import outcomes, tasks
from rsi.state import Layout
from tests.helpers import minimal_config

EVIDENCE = {"kind": "evidence", "id": "e1", "at": "2026-11-19T10:00:00Z",
            "summary": "Finding refuted by the author", "task_id": "t01"}
HELDOUT_EVIDENCE = {"kind": "evidence", "id": "e2", "summary": "About a held-out task", "task_id": "t07"}
TASK = {"kind": "task", "id": "live-1", "summary": "Worth replaying", "task": {"id": "t13", "text": "x"}}


class ParseTest(unittest.TestCase):
    def test_valid_records(self):
        text = "\n".join(json.dumps(r) for r in (EVIDENCE, TASK)) + "\n\n"
        records, problems = outcomes.parse_records(text)
        self.assertEqual(records, [EVIDENCE, TASK])
        self.assertEqual(problems, [])

    def test_bad_lines_are_skipped_and_reported(self):
        lines = [
            "starting export...",
            json.dumps(EVIDENCE),
            json.dumps({"kind": "rumour", "id": "x", "summary": "s"}),
            json.dumps({"kind": "evidence", "summary": "no id"}),
            json.dumps({"kind": "evidence", "id": "x"}),
            json.dumps({"kind": "task", "id": "x", "summary": "s"}),
            json.dumps({"kind": "task", "id": "x", "summary": "s", "task": {"id": "bad id"}}),
            json.dumps({"kind": "evidence", "id": "x", "summary": "s", "task_id": 7}),
            "[1]",
        ]
        records, problems = outcomes.parse_records("\n".join(lines))
        self.assertEqual(records, [EVIDENCE])
        self.assertEqual([p.split(":")[0] for p in problems],
                         ["line 1", "line 3", "line 4", "line 5", "line 6", "line 7", "line 8", "line 9"])


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(self.tmp / "home")}))
        self.layout = Layout("toy")
        self.project = self.tmp / "project"
        self.project.mkdir()
        self.feed = self.tmp / "feed.jsonl"
        self.since_log = self.tmp / "since.log"
        command = f"echo {{since}} >> {shlex.quote(str(self.since_log))}; cat {shlex.quote(str(self.feed))}"
        self.cfg = minimal_config(ingest=command)
        tasks_path = self.layout.tasks_path(None)
        tasks_path.parent.mkdir(parents=True)
        tasks_path.write_text('{"id": "t01"}\n{"id": "t07"}\n')

    def feed_records(self, *records):
        self.feed.write_text("".join(json.dumps(r) + "\n" for r in records))

    def run_ingest(self, when="2026-11-20T09:00:00Z"):
        now = datetime.strptime(when, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return outcomes.ingest(self.cfg, self.layout, self.project, now=now)

    def test_stores_evidence_and_queues_tasks(self):
        self.feed_records(EVIDENCE, TASK)
        self.assertEqual(self.run_ingest(), {"evidence": 1, "tasks": 1, "skipped": []})
        self.assertEqual(json.loads(self.layout.evidence_file.read_text()), EVIDENCE)
        self.assertEqual(tasks.pending_count(self.layout), 1)

    def test_records_seen_before_are_ignored(self):
        self.feed_records(EVIDENCE, TASK)
        self.run_ingest()
        self.assertEqual(self.run_ingest(), {"evidence": 0, "tasks": 0, "skipped": []})
        self.assertEqual(len(self.layout.evidence_file.read_text().splitlines()), 1)

    def test_a_task_already_in_the_benchmark_is_not_queued(self):
        self.feed_records({"kind": "task", "id": "live-2", "summary": "s", "task": {"id": "t01"}})
        self.assertEqual(self.run_ingest()["tasks"], 0)

    def test_since_is_empty_first_then_the_previous_ingest_time(self):
        self.feed_records(EVIDENCE)
        self.run_ingest("2026-11-20T09:00:00Z")
        self.run_ingest("2026-11-21T09:00:00Z")
        self.assertEqual(self.since_log.read_text().splitlines(), ["", "2026-11-20T09:00:00Z"])

    def test_no_command_is_a_no_op(self):
        result = outcomes.ingest(minimal_config(), self.layout, self.project)
        self.assertEqual(result, {"evidence": 0, "tasks": 0, "skipped": []})

    def test_a_failing_command_raises_and_keeps_the_old_stamp(self):
        cfg = minimal_config(ingest="echo broken >&2; exit 5")
        with self.assertRaises(outcomes.OutcomeError) as caught:
            outcomes.ingest(cfg, self.layout, self.project)
        self.assertIn("exit code 5", str(caught.exception))
        self.assertFalse(self.layout.ingest_stamp.exists())

    def test_skipped_lines_are_reported(self):
        self.feed.write_text("noise\n" + json.dumps(EVIDENCE) + "\n")
        result = self.run_ingest()
        self.assertEqual(result["evidence"], 1)
        self.assertEqual(len(result["skipped"]), 1)


class EvidenceCursorTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.dict(os.environ, {"RSI_HOME": str(tmp)}))
        self.layout = Layout("toy")
        self.layout.outcomes.mkdir(parents=True)

    def add(self, *records):
        with self.layout.evidence_file.open("a") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

    def test_each_record_is_new_once(self):
        self.assertEqual(outcomes.new_evidence(self.layout, set()), ([], 0))
        self.add(EVIDENCE)
        records, cursor = outcomes.new_evidence(self.layout, set())
        self.assertEqual((records, cursor), ([EVIDENCE], 1))
        self.assertEqual(outcomes.new_evidence(self.layout, set())[0], [EVIDENCE])
        outcomes.mark_shown(self.layout, cursor)
        self.assertEqual(outcomes.new_evidence(self.layout, set()), ([], 1))

    def test_evidence_about_heldout_tasks_is_withheld(self):
        self.add(EVIDENCE, HELDOUT_EVIDENCE)
        records, cursor = outcomes.new_evidence(self.layout, {"t07"})
        self.assertEqual((records, cursor), ([EVIDENCE], 2))
