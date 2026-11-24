import os
import shlex
import signal
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rsi import evaluator
from tests.helpers import minimal_config

# A run_task command for tests. Usage: runner.py <task_file> <out>
RUNNER = r'''
import json, os, sys, time
task = json.load(open(sys.argv[1]))
out = sys.argv[2]
log = os.environ.get("CALL_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(task["id"] + " " + os.environ["RSI_ROLLOUT"] + " " + os.path.basename(os.getcwd()) + "\n")
mode = task.get("mode", "ok")
if mode == "fail":
    sys.exit(2)
if mode == "sleep":
    time.sleep(30)
if mode == "none":
    sys.exit(0)
if mode == "dir":
    os.mkdir(os.path.join(out, "result.json"))
    sys.exit(0)
if mode == "raw":
    with open(os.path.join(out, "result.json"), "w") as fh:
        fh.write(task["raw"])
    sys.exit(0)
names = ("RSI_TASK_ID", "RSI_ROLLOUT", "RSI_MODEL", "RSI_BENCHMARK")
summary = sys.argv[1] if mode == "where" else " ".join(os.environ[n] for n in names)
if mode == "current":
    summary = os.environ.get("RSI_CURRENT", "unset") or "empty"
score = task["scores"][int(os.environ["RSI_ROLLOUT"]) - 1] if "scores" in task else task["score"]
result = {"score": score, "summary": summary}
metrics = task.get("metrics")
if isinstance(metrics, list):
    metrics = metrics[int(os.environ["RSI_ROLLOUT"]) - 1]
if metrics is not None:
    result["metrics"] = metrics
with open(os.path.join(out, "result.json"), "w") as fh:
    json.dump(result, fh)
print("ran", task["id"])
'''


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.checkout = self.tmp / "check out"
        self.checkout.mkdir()
        (self.checkout / "runner.py").write_text(RUNNER)
        self.results = self.tmp / "results dir"
        self.call_log = self.tmp / "calls.log"
        self.home = self.tmp / "home"
        self.enterContext(mock.patch.dict(os.environ, {"CALL_LOG": str(self.call_log), "RSI_HOME": str(self.home)}))
        self.cfg = minimal_config(
            run_task=f"{shlex.quote(sys.executable)} runner.py {{task_file}} {{out}}",
            rollouts=2,
            task_timeout_s=20,
        )

    def run_tasks(self, task_list, cfg=None):
        return evaluator.evaluate(cfg or self.cfg, task_list, self.checkout, self.results, "bench-v1")

    def test_scores_are_averaged(self):
        res = self.run_tasks([{"id": "a", "score": 1}, {"id": "b", "score": 0.5}, {"id": "c", "score": 0}])
        self.assertEqual(res.task_scores(), {"a": 1.0, "b": 0.5, "c": 0.0})
        self.assertEqual(res.score(), 50.0)
        self.assertEqual(res.passing(0.5), {"a", "b"})
        self.assertEqual(res.passed_total(0.5), (4, 6))
        self.assertEqual(res.infra_failure_rate(), 0.0)

    def test_task_score_is_the_mean_of_its_rollouts(self):
        res = self.run_tasks([{"id": "a", "scores": [1, 0]}, {"id": "b", "scores": [0.5, 1]}])
        self.assertEqual(res.task_scores(), {"a": 0.5, "b": 0.75})
        self.assertEqual(res.score(), 62.5)
        self.assertEqual(res.passed_total(0.5), (3, 4))

    def test_task_file_is_kept_in_the_private_directory_and_removed(self):
        res = self.run_tasks([{"id": "a", "score": 1, "mode": "where"}], replace(self.cfg, rollouts=1))
        task_file = Path(res.rollouts[0].summary)
        self.assertTrue(task_file.is_relative_to(self.home / "toy" / "private"), task_file)
        self.assertFalse(task_file.exists())

    def test_score_has_two_decimals(self):
        res = self.run_tasks([{"id": "a", "score": 1}, {"id": "b", "score": 0}, {"id": "c", "score": 0}])
        self.assertEqual(res.score(), 33.33)

    def test_command_failure_scores_zero_and_stays_in_the_denominator(self):
        res = self.run_tasks([{"id": "a", "score": 1}, {"id": "b", "mode": "fail"}])
        self.assertEqual(res.score(), 50.0)
        self.assertEqual(res.infra_failure_rate(), 0.5)
        self.assertEqual({r.task_id: r.error for r in res.rollouts}, {"a": None, "b": "exit code 2"})

    def test_invalid_result_is_an_infra_failure(self):
        raws = {
            "nan": '{"score": NaN}',
            "bool": '{"score": true}',
            "text": '{"score": "1"}',
            "big": '{"score": 1.5}',
            "neg": '{"score": -0.1}',
            "list": "[1]",
            "junk": "not json",
            "huge": '{"score": 1' + "0" * 400 + "}",
        }
        task_list = [{"id": k, "mode": "raw", "raw": v} for k, v in raws.items()]
        task_list.append({"id": "missing", "mode": "none"})
        task_list.append({"id": "folder", "mode": "dir"})
        res = self.run_tasks(task_list, replace(self.cfg, rollouts=1))
        self.assertEqual(res.score(), 0.0)
        self.assertEqual(res.infra_failure_rate(), 1.0)
        errors = {r.task_id: r.error for r in res.rollouts}
        self.assertEqual(errors["folder"], "result.json could not be read")
        for name in ("nan", "bool", "text", "big", "neg", "huge"):
            self.assertEqual(errors[name], "score must be a number from 0 to 1", name)
        self.assertEqual(errors["list"], "result.json is not an object")
        self.assertEqual(errors["junk"], "result.json is not JSON")
        self.assertEqual(errors["missing"], "result.json missing")

    def test_timeout_is_an_infra_failure(self):
        res = self.run_tasks([{"id": "slow", "mode": "sleep"}], replace(self.cfg, rollouts=1, task_timeout_s=1))
        self.assertEqual(res.rollouts[0].error, "timeout after 1s")
        self.assertEqual(res.rollouts[0].score, 0.0)

    def test_an_interrupt_stops_running_commands_and_records_nothing(self):
        timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT))  # what Ctrl-C sends
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(KeyboardInterrupt):
                self.run_tasks([{"id": "slow1", "mode": "sleep"}, {"id": "slow2", "mode": "sleep"}],
                               replace(self.cfg, rollouts=1))
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 10)  # the commands would otherwise run for 20s
        self.assertEqual(list(self.results.rglob("rollout.json")), [])

    def test_finished_rollouts_are_not_run_again(self):
        task_list = [{"id": "a", "score": 1}]
        first = self.run_tasks(task_list)
        second = self.run_tasks(task_list)
        self.assertEqual(first.rollouts, second.rollouts)
        self.assertEqual(len(self.call_log.read_text().splitlines()), 2)

    def test_half_finished_rollout_is_run_again(self):
        stale = self.results / "a" / "r1"
        stale.mkdir(parents=True)
        (stale / "result.json").write_text('{"score": 0}')
        (stale / "junk").write_text("left by a killed run")
        res = self.run_tasks([{"id": "a", "score": 1}])
        self.assertEqual(res.score(), 100.0)
        self.assertFalse((stale / "junk").exists())

    def test_paths_with_spaces(self):
        self.assertIn(" ", str(self.checkout))
        self.assertIn(" ", str(self.results))
        res = self.run_tasks([{"id": "a", "score": 1}])
        self.assertEqual(res.score(), 100.0)
        self.assertTrue((self.results / "a" / "r1" / "result.json").is_file())

    def test_environment_and_output_files(self):
        res = self.run_tasks([{"id": "a", "score": 1}], replace(self.cfg, rollouts=1, model="model-x"))
        self.assertEqual(res.rollouts[0].summary, "a 1 model-x bench-v1")
        out = self.results / "a" / "r1"
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            ["result.json", "rollout.json", "stderr.log", "stdout.log"],
        )
        self.assertEqual((out / "stdout.log").read_text().strip(), "ran a")

    def test_table(self):
        res = self.run_tasks([{"id": "a", "score": 1}, {"id": "b", "mode": "fail"}], replace(self.cfg, rollouts=1))
        self.assertEqual(
            res.table(0.5),
            [
                {"task": "a", "score": 1.0, "passes": True, "errors": 0, "summary": "a 1 model-a bench-v1", "metrics": {}},
                {"task": "b", "score": 0.0, "passes": False, "errors": 1, "summary": "exit code 2", "metrics": {}},
            ],
        )

    def test_evaluate_many_keeps_targets_apart(self):
        other = self.tmp / "other"
        other.mkdir()
        (other / "runner.py").write_text(RUNNER.replace('{"score": score,', '{"score": 0,'))
        targets = {"x": (self.checkout, self.results / "x"), "y": (other, self.results / "y")}
        results = evaluator.evaluate_many(self.cfg, [{"id": "a", "score": 1}], targets, "bench-v1")
        self.assertEqual({name: r.score() for name, r in results.items()}, {"x": 100.0, "y": 0.0})
        self.assertTrue((self.results / "y" / "a" / "r2" / "rollout.json").is_file())

    def test_metrics_are_read_and_averaged(self):
        res = self.run_tasks([
            {"id": "a", "score": 1, "metrics": [{"wall_s": 600, "cost": 5.0}, {"wall_s": 700, "cost": 6.0}]},
            {"id": "b", "score": 1, "metrics": {"wall_s": 100}},
        ])
        self.assertEqual(res.task_metrics(), {"a": {"wall_s": 650.0, "cost": 5.5}, "b": {"wall_s": 100.0}})
        self.assertEqual(res.metric_means(), {"wall_s": 375.0, "cost": 5.5})
        self.assertEqual(res.table(0.5)[0]["metrics"], {"wall_s": 650.0, "cost": 5.5})
        self.assertEqual(res.rollouts[0].metrics, {"wall_s": 600.0, "cost": 5.0})

    def test_metric_means_skip_rollouts_without_the_metric(self):
        res = self.run_tasks([{"id": "a", "score": 1, "metrics": [{"accuracy_pct": 90.00}, None]}])
        self.assertEqual(res.metric_means(), {"accuracy_pct": 90.00})

    def test_failed_rollouts_report_no_metrics(self):
        res = self.run_tasks([{"id": "a", "mode": "fail"}])
        self.assertEqual((res.rollouts[0].metrics, res.task_metrics(), res.metric_means()), ({}, {}, {}))

    def test_invalid_metrics_are_an_infra_failure(self):
        raws = ['{"score": 1, "metrics": [1]}', '{"score": 1, "metrics": {"Wall": 1}}',
                '{"score": 1, "metrics": {"wall_s": "600"}}', '{"score": 1, "metrics": {"wall_s": true}}',
                '{"score": 1, "metrics": {"wall_s": NaN}}', '{"score": 1, "metrics": {"wall_s": 1e999}}',
                '{"score": 1, "metrics": {"wall_s": 1' + "0" * 400 + "}}"]
        task_list = [{"id": f"m{i}", "mode": "raw", "raw": raw} for i, raw in enumerate(raws)]
        res = self.run_tasks(task_list, replace(self.cfg, rollouts=1))
        for r in res.rollouts:
            self.assertEqual((r.score, r.error, r.metrics),
                             (0.0, "metrics must map names like wall_s to finite numbers", {}), r.task_id)

    def test_old_marker_without_metrics_loads(self):
        marker = self.results / "a" / "r1" / "rollout.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"task_id": "a", "rollout": 1, "score": 1.0, "error": null, "summary": ""}')
        res = self.run_tasks([{"id": "a", "score": 0}], replace(self.cfg, rollouts=1))
        self.assertEqual((res.score(), res.rollouts[0].metrics), (100.0, {}))
        self.assertFalse(self.call_log.exists())

    def test_the_current_version_is_marked_for_the_scorer(self):
        x, y = self.tmp / "x", self.tmp / "y"
        for folder in (x, y):
            folder.mkdir()
            (folder / "runner.py").write_text(RUNNER)
        targets = {"x": (x, self.results / "x"), "y": (y, self.results / "y")}
        results = evaluator.evaluate_many(replace(self.cfg, rollouts=1), [{"id": "a", "score": 1, "mode": "current"}],
                                          targets, "bench-v1", current={"x"})
        self.assertEqual({name: r.rollouts[0].summary for name, r in results.items()}, {"x": "1", "y": "empty"})
        only = evaluator.evaluate(replace(self.cfg, rollouts=1), [{"id": "b", "score": 1, "mode": "current"}],
                                  x, self.results / "z", "bench-v1", current=True)
        self.assertEqual(only.rollouts[0].summary, "1")

    def test_on_done_reports_each_run_and_whether_it_was_cached(self):
        seen = []
        task_list = [{"id": "a", "score": 1}, {"id": "b", "mode": "fail"}]
        evaluator.evaluate_many(self.cfg, task_list, {"x": (self.checkout, self.results)}, "bench-v1",
                                on_done=lambda target, rollout, cached: seen.append((target, rollout, cached)))
        self.assertEqual(sorted((t, r.task_id, r.rollout, c) for t, r, c in seen),
                         [("x", "a", 1, False), ("x", "a", 2, False), ("x", "b", 1, False), ("x", "b", 2, False)])
        self.assertTrue(all(r.seconds > 0 for _, r, _ in seen))
        again = []
        evaluator.evaluate_many(self.cfg, task_list, {"x": (self.checkout, self.results)}, "bench-v1",
                                on_done=lambda target, rollout, cached: again.append(cached))
        self.assertEqual(again, [True] * 4)

    def test_on_start_is_called_before_each_run_that_really_runs(self):
        events, lock = [], threading.Lock()
        def log(kind):
            def record(target, *rest):
                with lock:
                    events.append((kind, target))
            return record
        task_list = [{"id": "a", "score": 1}, {"id": "b", "score": 0}]
        evaluator.evaluate_many(self.cfg, task_list, {"x": (self.checkout, self.results)}, "bench-v1",
                                on_start=log("start"), on_done=log("done"))
        self.assertEqual(sorted(events), [("done", "x")] * 4 + [("start", "x")] * 4)
        for n in range(1, 5):   # by any point, at least as many runs have started as have finished
            prefix = events[:n]
            self.assertGreaterEqual(prefix.count(("start", "x")), prefix.count(("done", "x")))
        events.clear()
        evaluator.evaluate_many(self.cfg, task_list, {"x": (self.checkout, self.results)}, "bench-v1",
                                on_start=log("start"), on_done=log("done"))
        self.assertEqual(events, [("done", "x")] * 4)        # all read back from disk: none started

    def test_old_marker_has_zero_seconds(self):
        marker = self.results / "a" / "r1" / "rollout.json"
        marker.parent.mkdir(parents=True)
        marker.write_text('{"task_id": "a", "rollout": 1, "score": 1.0, "error": null, "summary": ""}')
        res = self.run_tasks([{"id": "a", "score": 0}], replace(self.cfg, rollouts=1))
        self.assertEqual(res.rollouts[0].seconds, 0.0)

    def test_rollouts_are_submitted_round_robin(self):
        x, y = self.tmp / "x", self.tmp / "y"
        for folder in (x, y):
            folder.mkdir()
            (folder / "runner.py").write_text(RUNNER)
        targets = {"x": (x, self.results / "x"), "y": (y, self.results / "y")}
        evaluator.evaluate_many(replace(self.cfg, parallel=1), [{"id": "a", "score": 1}], targets, "bench-v1")
        order = [tuple(line.split()[1:]) for line in self.call_log.read_text().splitlines()]
        self.assertEqual(order, [("1", "x"), ("1", "y"), ("2", "x"), ("2", "y")])

    def test_no_tasks_scores_zero(self):
        res = self.run_tasks([])
        self.assertEqual((res.score(), res.infra_failure_rate(), res.passed_total(0.5)), (0.0, 0.0, (0, 0)))
