from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from alphatransfer.config import V0Config
from alphatransfer.evaluation import fit_live, walk_forward
from alphatransfer.features import build_feature_rows, feature_names, local_minimum_label, window_closing_label
from alphatransfer.market_calendar import calendar_features
from alphatransfer.normalization import cbr_rub_per_unit, moex_rub_per_kzt, nbk_rub_per_kzt
from alphatransfer.policy import apply_policy
from alphatransfer.reporting import build_snapshot, render_report
from alphatransfer.schema import Observation


UTC = timezone.utc


def obs(day: date, source: str, symbol: str, field: str, raw: float, nominal: float, normalized: float, unit: str) -> Observation:
    return Observation(day, datetime.combine(day, datetime.min.time(), UTC), source, symbol, field, nominal, raw, normalized, unit)


def synthetic(days: int = 800) -> list[Observation]:
    rows = []
    start = date(2020, 1, 2)
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        phase = i % 31
        cbr = .19 - min(phase, 20) * .0006 + max(0, phase - 20) * .0009
        nbk = cbr * (1 + .0015)
        moex = cbr * (1 - .001)
        rows.extend([
            obs(day, "CBR", "KZT", "close", cbr * 100, 100, cbr, "RUB_per_KZT"),
            obs(day, "CBR", "USD", "close", 75, 1, 75, "RUB_per_USD"),
            obs(day, "CBR", "CNY", "close", 11, 1, 11, "RUB_per_CNY"),
            obs(day, "NBK", "RUB", "close", 1 / nbk, 1, nbk, "RUB_per_KZT"),
            obs(day, "NBK", "USD", "close", 75 / nbk, 1, 75 / nbk, "KZT_per_USD"),
            obs(day, "NBK", "CNY", "close", 11 / nbk, 1, 11 / nbk, "KZT_per_CNY"),
        ])
        for field, value in (("open", moex * .999), ("high", moex * 1.004), ("low", moex * .996), ("close", moex)):
            rows.append(obs(day, "MOEX", "KZTRUB_TOM", field, value * 100, 100, value, "RUB_per_KZT"))
    return rows


class KztV0Test(TestCase):
    def test_all_source_directions_and_nominals(self):
        self.assertAlmostEqual(cbr_rub_per_unit(19.5, 100), .195)
        self.assertAlmostEqual(nbk_rub_per_kzt(5.0, 1), .2)
        self.assertAlmostEqual(moex_rub_per_kzt(19.7, 100), .197)

    def test_moex_high_low_preserve_direction(self):
        rows = build_feature_rows(synthetic(8), V0Config.load().section("features"))
        self.assertGreater(rows[0]["moex_high"], rows[0]["moex_close"])
        self.assertLess(rows[0]["moex_low"], rows[0]["moex_close"])

    def test_fill_does_not_create_momentum(self):
        data = synthetic(8)
        stale_day = date(2020, 1, 7)
        data = [r for r in data if not (r.source == "CBR" and r.effective_date == stale_day)]
        rows = build_feature_rows(data, V0Config.load().section("features"))
        stale = next(r for r in rows if r["date"] == stale_day.isoformat())
        self.assertEqual(stale["cbr_is_fresh"], 0)
        self.assertEqual(stale["decreasing_run"], 0)
        self.assertTrue(all(stale[f"return_{n}"] == 0 for n in (1, 3, 5, 10)))

    def test_both_holiday_calendars_and_freshness_age(self):
        self.assertEqual(calendar_features(date(2025, 6, 12))["is_ru_holiday"], 1)
        self.assertEqual(calendar_features(date(2025, 3, 21))["is_kz_holiday"], 1)
        data = synthetic(8)
        stale_day = date(2020, 1, 7)
        data = [r for r in data if not (r.source == "CBR" and r.effective_date == stale_day)]
        row = next(r for r in build_feature_rows(data, V0Config.load().section("features")) if r["date"] == stale_day.isoformat())
        self.assertGreater(row["cbr_age_days"], 0)

    def test_features_are_unchanged_by_future_data(self):
        data = synthetic(90); cutoff = date(2020, 2, 20)
        prefix = [r for r in data if r.effective_date <= cutoff]
        a = build_feature_rows(prefix, V0Config.load().section("features"))
        b = [r for r in build_feature_rows(data, V0Config.load().section("features")) if r["date"] <= cutoff.isoformat()]
        self.assertEqual(a, b)

    def test_scenarios_have_distinct_truth(self):
        rows = [{"cbr_rate": x, "cbr_is_fresh": 1} for x in (3, 2, 1, 1.02, 1.08, 1.12, 1.2)]
        self.assertEqual(local_minimum_label(rows, 2, 2), 1)
        self.assertEqual(window_closing_label(rows, 4, 3, 500), 1)
        self.assertEqual(local_minimum_label(rows, 4, 2), 0)

    def test_live_model_does_not_change_when_future_rows_are_appended(self):
        rows = build_feature_rows(synthetic(700), V0Config.load().section("features"))
        cutoff = date.fromisoformat(rows[-30]["date"])
        selection = {"feature_names": feature_names(rows), "last_selection": {"selected_model": "logistic_regression", "selected_parameter": 1.0, "threshold": .65, "rebound_threshold_bps": 20.0}}
        config = V0Config.load().raw
        model_a, _ = fit_live([r for r in rows if r["date"] <= cutoff.isoformat()], cutoff, selection, config)
        model_b, _ = fit_live(rows, cutoff, selection, config)
        probe = next(r for r in rows if r["date"] == cutoff.isoformat())
        self.assertEqual(model_a.weights, model_b.weights)
        self.assertEqual(model_a.predict(probe), model_b.predict(probe))

    def test_cooldown_and_weekly_cap(self):
        candidates = [{"index": i, "date": (date(2025, 1, 1) + timedelta(days=i)).isoformat(), "scenario": "favorable_now", "confidence": .9, "evidence": [], "rate": .2} for i in (0, 1, 4, 6)]
        result = apply_policy(candidates, cooldown_sessions=3, cap_7d=2)
        self.assertTrue(result[0]["policy_eligible"])
        self.assertEqual(result[1]["suppressed_reason"], "cooldown")
        self.assertTrue(result[2]["policy_eligible"])
        self.assertEqual(result[3]["suppressed_reason"], "cooldown")

    def test_report_is_reproducible(self):
        rows = build_feature_rows(synthetic(30), V0Config.load().section("features"))
        bt = {"primary_lifts": {"favorable_now": 1.0, "window_closing": 0.0}, "metrics": [], "source_robustness": [], "events": [], "frequency": {}}
        signal = {"scenario": None, "eligible_to_send": False, "confidence": 0, "rate_snapshot": {}, "evidence": [], "suppressed_reason": "no_candidate"}
        snapshot = build_snapshot(rows, bt, signal)
        self.assertEqual(render_report(snapshot, rows), render_report(deepcopy(snapshot), deepcopy(rows)))
