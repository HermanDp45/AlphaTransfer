from __future__ import annotations

from datetime import date, timedelta
from unittest import TestCase
from unittest.mock import patch

from alphatransfer.backtest import evaluate_events, walk_forward_backtest, walk_forward_ml_backtest
from alphatransfer.calendar import calendar_daily
from alphatransfer.cbr import Quote
from alphatransfer.engine import SignalConfig, indicator_rows, signal_at, signals_as_of
from alphatransfer.cbr import fetch_cbr_daily_history
from alphatransfer.ml import LocalMinimumLogit, ml_signal_at


def series(length: int = 430, corridor: str = "TJS") -> list[Quote]:
    """Deterministic publication-day path with down legs and recoveries."""
    start = date(2023, 1, 2)
    quotes = []
    for i in range(length):
        phase = i % 28
        # 19 day decline followed by a 9 day recovery.
        value = 0.11 - phase * 0.001 if phase < 19 else 0.091 + (phase - 19) * 0.0014
        quotes.append(Quote(start + timedelta(days=i), corridor, value, 1, value))
    return quotes


class SignalLayerTest(TestCase):
    def setUp(self) -> None:
        self.quotes = series()
        self.config = SignalConfig(lookback=20, momentum_days=3, low_percentile=.3, rebound_window=10, cooldown_observations=3)

    def test_calendar_marks_forward_fills_as_not_observed(self) -> None:
        data = [Quote(date(2024, 1, 5), "AMD", .0025, 100, .25), Quote(date(2024, 1, 8), "AMD", .0026, 100, .26)]
        daily = calendar_daily(data)
        self.assertEqual(len(daily), 4)
        self.assertFalse(daily[1]["is_publication_day"])
        self.assertEqual(daily[1]["rub_per_unit"], .0025)

    def test_indicators_do_not_count_calendar_flat_days(self) -> None:
        features = indicator_rows(self.quotes, self.config)
        self.assertEqual(len(features), len(self.quotes))
        self.assertTrue(any(row["momentum_down"] for row in features))
        self.assertTrue(any(row["rebound_up"] for row in features))

    def test_signal_at_ignores_all_future_input(self) -> None:
        as_of = self.quotes[150].date
        expected = signal_at(self.quotes, as_of, self.config)
        future_mutated = self.quotes + [Quote(as_of + timedelta(days=200), "TJS", .00001, 1, .00001)]
        self.assertEqual(expected, signal_at(future_mutated, as_of, self.config))

    def test_thinning_uses_publication_observation_distance(self) -> None:
        events = signals_as_of(self.quotes, self.quotes[-1].date, self.config)
        by_key: dict[tuple[str, str], list[date]] = {}
        for event in events:
            by_key.setdefault((event["corridor"], event["direction"]), []).append(event["date"])
        for dates in by_key.values():
            indexes = [next(i for i, q in enumerate(self.quotes) if q.date == d) for d in dates]
            self.assertTrue(all(b - a > self.config.cooldown_observations for a, b in zip(indexes, indexes[1:])))

    def test_metrics_have_random_baseline_and_benefit(self) -> None:
        events = signals_as_of(self.quotes, self.quotes[-1].date, self.config)
        report = evaluate_events(self.quotes, events, horizon=3)
        self.assertTrue(report["metrics"])
        row = report["metrics"][0]
        self.assertIn("random_day_hit_rate", row)
        self.assertIn("mean_benefit_bps", row)

    def test_walk_forward_selects_only_prior_configurations(self) -> None:
        configs = [self.config, SignalConfig(lookback=30, momentum_days=4, low_percentile=.25, rebound_window=12, cooldown_observations=4)]
        report = walk_forward_backtest(self.quotes, configs, train_observations=100, test_observations=60, horizon=3)
        self.assertTrue(report["folds"])
        self.assertTrue(all("chosen_config" in fold for fold in report["folds"]))

    def test_cbr_xml_nominal_is_normalized_to_one_currency_unit(self) -> None:
        payload = b'<?xml version="1.0"?><ValCurs><Record Date="02.01.2024"><Nominal>100</Nominal><Value>25,0</Value></Record></ValCurs>'
        class Response:
            def read(self): return payload
            def __enter__(self): return self
            def __exit__(self, *_): return False
        with patch("alphatransfer.cbr.urllib.request.urlopen", return_value=Response()):
            quotes = fetch_cbr_daily_history(["AMD"], "2024-01-01", "2024-01-03")
        self.assertEqual(quotes[0].nominal, 100)
        self.assertEqual(quotes[0].rub_per_unit, .25)

    def test_classical_model_uses_only_resolved_labels_before_cutoff(self) -> None:
        cutoff = self.quotes[300].date
        model = LocalMinimumLogit(horizon=3, epochs=80).fit(self.quotes, cutoff, self.config)
        future = self.quotes + [Quote(cutoff + timedelta(days=50), "TJS", .000001, 1, .000001)]
        same_model = LocalMinimumLogit(horizon=3, epochs=80).fit(future, cutoff, self.config)
        self.assertEqual(model.weights, same_model.weights)
        events = ml_signal_at(self.quotes, cutoff, model, self.config)
        self.assertTrue(all(event["date"] == cutoff for event in events))

    def test_classical_model_has_out_of_time_walk_forward_path(self) -> None:
        report = walk_forward_ml_backtest(self.quotes, self.config, train_observations=100, test_observations=60, horizon=3)
        self.assertTrue(report["folds"])
        self.assertTrue(all(fold["model"]["training_rows"] > 0 for fold in report["folds"]))
