import unittest

from rsi import progress


class FormatTest(unittest.TestCase):
    def test_duration(self):
        self.assertEqual([progress.duration(s) for s in (0, 45, 59.9, 60, 2280, 3600, 15120)],
                         ["0s", "45s", "59s", "1m", "38m", "1h00m", "4h12m"])

    def test_default_template_is_valid(self):
        progress.check_template(progress.DEFAULT_LINE)

    def test_unknown_fields_are_refused(self):
        for line in ("{rnd}", "{metric.Wall}", "{limit.}", "{metric}", "{round.x}", "{}", "{0}"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                progress.check_template(line)
        progress.check_template("{metric.wall_s:>8} {limit.cost_usd} {round!s} plain text {{braces}}")

    def test_a_broken_template_is_refused(self):
        with self.assertRaises(ValueError):
            progress.check_template("{round")

    def test_render(self):
        snap = progress.Snapshot(run="r1", status="running", round=2, rounds=5, step="measuring", done=9, total=16,
                                 failed=1, agents_running=0, agents_total=0, errors=3, accepted=1, score=61.2,
                                 start_score=50.0, metrics={"wall_s": 636.0}, limits={"cost_usd": 6.0},
                                 elapsed=15120, step_elapsed=2280, eta=25500)
        line = progress.render("round {round}/{rounds} · {step} · runs {done}/{total}, {failed} failed · agents {agents} · "
                               "score {score} (start {start_score}) · {metric.wall_s} · {limit.cost_usd} · "
                               "{step_elapsed} in step, {elapsed} total, ~{eta} left · {errors} errors, {accepted} kept", snap)
        self.assertEqual(line, "round 2/5 · measuring · runs 9/16, 1 failed · agents - · score 61.20 (start 50.00) · "
                               "636.00 · +6.00% · 38m in step, 4h12m total, ~7h05m left · 3 errors, 1 kept")

    def test_agents_running(self):
        snap = progress.Snapshot(run="r1", agents_running=2, agents_total=3)
        self.assertEqual(progress.render("{agents} {run} {status}", snap), "2/3 r1 running")

    def test_missing_values_render_as_a_dash(self):
        snap = progress.Snapshot(run="r1")
        self.assertEqual(progress.render("{score} {metric.x} {limit.y} {eta} {agents}", snap), "- - - ? -")

    def test_default_line_renders(self):
        line = progress.render(progress.DEFAULT_LINE, progress.Snapshot(run="r1", round=1, rounds=5, step="baseline"))
        self.assertIn("round 1/5 · baseline · runs 0/0, 0 failed", line)


class EtaTest(unittest.TestCase):
    def test_eta(self):
        self.assertEqual(progress.eta_seconds(16, [600, 900], 4, 2, [600]), 16 * 750 / 4 + 2 * 600)
        self.assertEqual(progress.eta_seconds(16, [600], 4, 2, []), 2400)
        self.assertIsNone(progress.eta_seconds(16, [], 4, 2, [600]))
        self.assertEqual(progress.eta_seconds(0, [600], 4, 0, [600]), 0)
