import unittest

from rsi import measure
from rsi.config import Constraint

C = (Constraint("cost_usd", "increase", 10.0), Constraint("accuracy_pct", "decrease", 0.0))


class PctChangeTest(unittest.TestCase):
    def test_pct_change(self):
        self.assertEqual(measure.pct_change(4.00, 4.24), 6.00)
        self.assertEqual(measure.pct_change(800, 600), -25.00)
        self.assertEqual(measure.pct_change(5, 5), 0.0)
        self.assertIsNone(measure.pct_change(0, 5))

    def test_changes_cover_metrics_on_both_sides(self):
        got = measure.changes({"a": 2.0, "b": 4.0, "zero": 0.0, "only_before": 1.0}, {"a": 4.0, "b": 4.0, "zero": 3.0, "only_after": 1.0})
        self.assertEqual(got, {"a": 100.0, "b": 0.0})


class LimitStatusTest(unittest.TestCase):
    def test_within_and_over(self):
        ok = measure.limit_status(C, {}, {"cost_usd": 4.00, "accuracy_pct": 90.00}, {"cost_usd": 4.24, "accuracy_pct": 90.00})
        self.assertEqual([s["state"] for s in ok], ["ok", "ok"])
        self.assertEqual(ok[0], {"metric": "cost_usd", "direction": "increase", "allowed_pct": 10.0,
                                 "total_pct": 6.00, "state": "ok"})
        self.assertEqual(measure.blocking(ok), [])
        over = measure.limit_status(C, {"cost_usd": 1.06}, {"cost_usd": 5.0, "accuracy_pct": 90.00},
                                    {"cost_usd": 5.25, "accuracy_pct": 89.99})
        self.assertEqual([(s["state"], s["total_pct"]) for s in over], [("over", 11.3), ("over", -0.01)])
        self.assertEqual(measure.blocking(over), over)
        self.assertEqual(measure.describe(over[0]), "cost_usd +11.30% of +10.00% allowed")
        self.assertEqual(measure.describe(over[1]), "accuracy_pct -0.01% of -0.00% allowed")
        self.assertEqual(measure.describe(ok[0]), "cost_usd +6.00% of +10.00% allowed")

    def test_exactly_at_a_limit_is_ok(self):
        s = measure.limit_status(C, {}, {"cost_usd": 100, "accuracy_pct": 90.00}, {"cost_usd": 110, "accuracy_pct": 90.00})
        self.assertEqual([(x["state"], x["total_pct"]) for x in s], [("ok", 10.0), ("ok", 0.0)])

    def test_a_rise_never_breaks_a_decrease_limit(self):
        s = measure.limit_status(C, {}, {"accuracy_pct": 90.00}, {"accuracy_pct": 90.71})
        self.assertEqual((s[1]["state"], s[1]["total_pct"]), ("ok", 0.79))

    def test_a_fall_never_breaks_an_increase_limit(self):
        s = measure.limit_status(C, {}, {"cost_usd": 5.0}, {"cost_usd": 1.0})
        self.assertEqual((s[0]["state"], s[0]["total_pct"]), ("ok", -80.0))

    def test_earlier_rounds_count(self):
        s = measure.limit_status(C, {"cost_usd": 1.08}, {"cost_usd": 5.0}, {"cost_usd": 5.1})
        self.assertEqual((s[0]["state"], s[0]["total_pct"]), ("over", 10.16))

    def test_missing_metric_blocks(self):
        s = measure.limit_status(C, {}, {"cost_usd": 5.0, "accuracy_pct": 90.00}, {"cost_usd": 5.0})
        self.assertEqual((s[1]["state"], s[1]["total_pct"]), ("missing", None))
        self.assertEqual(measure.blocking(s), [s[1]])
        self.assertEqual(measure.describe(s[1]), "accuracy_pct was not reported by the candidate")

    def test_zero_baseline_is_not_measured(self):
        for base in ({"cost_usd": 0, "accuracy_pct": 0.0}, {}):
            s = measure.limit_status(C, {}, base, {"cost_usd": 5.0, "accuracy_pct": 90.00})
            self.assertEqual([(x["state"], x["total_pct"]) for x in s], [("not measured", None), ("not measured", None)])
            self.assertEqual(measure.blocking(s), [])
            self.assertEqual(measure.describe(s[0]), "cost_usd not measured")

    def test_no_constraints(self):
        self.assertEqual(measure.limit_status((), {}, {"a": 1.0}, {"a": 2.0}), [])

    def test_used_status(self):
        s = measure.used_status(C, {"cost_usd": 1.06})
        self.assertEqual([measure.describe(x) for x in s], ["cost_usd +6.00% of +10.00% allowed",
                                                            "accuracy_pct +0.00% of -0.00% allowed"])
