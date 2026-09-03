#!/usr/bin/env python3
"""Unit tests for the hackathon baseline and its no-lookahead contract."""

from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import hackathon_baseline as baseline  # noqa: E402


def make_points(values: list[float]) -> list[baseline.RatePoint]:
    start = date(2024, 1, 1)
    return [
        baseline.RatePoint(
            day=start + timedelta(days=index),
            rate=value,
            published_at=datetime.combine(
                start + timedelta(days=index), datetime.max.time(), timezone.utc
            ),
        )
        for index, value in enumerate(values)
    ]


class HackathonBaselineTest(unittest.TestCase):
    # Проверяет, что дописывание будущего не меняет прошлые сигналы.
    def test_prefix_invariance_for_all_indicators(self) -> None:
        # Arrange
        values = [100.0 + (index % 17) - index * 0.03 for index in range(220)]
        full = make_points(values)
        cutoff = 170

        # Act
        for indicator in baseline.indicators():
            full_candidates = [
                candidate
                for candidate in indicator.detector(full)
                if candidate.index < cutoff
            ]
            prefix_candidates = indicator.detector(full[:cutoff])

            # Assert
            self.assertEqual(full_candidates, prefix_candidates, indicator.name)

    # Проверяет строгую трактовку «сейчас не хуже следующих h дней».
    def test_now_hit_requires_no_better_future_day(self) -> None:
        # Arrange
        points = make_points([10.0, 11.0, 9.0, 12.0])

        # Act / Assert
        self.assertFalse(baseline.strict_hit(points, 0, 3, "NOW_FAVORABLE"))
        self.assertTrue(baseline.strict_hit(points, 2, 1, "NOW_FAVORABLE"))

    # Проверяет, что перенесённый курс не считается новой точкой сигнала.
    def test_unchanged_day_is_not_updated(self) -> None:
        # Arrange
        points = make_points([10.0, 10.0, 10.1])

        # Act / Assert
        self.assertFalse(baseline.is_updated(points, 1))
        self.assertTrue(baseline.is_updated(points, 2))

    # Проверяет, что cooldown не допускает серию сигналов внутри окна.
    def test_cooldown_uses_calendar_days(self) -> None:
        # Arrange
        points = make_points([10.0 + index for index in range(8)])
        candidates = [baseline.Candidate(index, 1.0) for index in (1, 2, 4, 7)]

        # Act
        selected = baseline.apply_cooldown(points, candidates, cooldown_days=3)

        # Assert
        self.assertEqual([candidate.index for candidate in selected], [1, 4, 7])

    # Проверяет знак выгоды: меньший текущий курс должен давать положительные б.п.
    def test_benefit_sign_is_positive_for_local_low(self) -> None:
        # Arrange
        points = make_points([11.0, 10.0, 11.0])

        # Act
        benefit = baseline.symmetric_benefit_bps(points, 1, 1)

        # Assert
        self.assertGreater(benefit, 0.0)

    # Проверяет end-to-end срез реального ряда по дате.
    def test_real_data_as_of_excludes_future_rows_and_signals(self) -> None:
        # Arrange
        cutoff = date(2025, 12, 31)
        path = PROJECT_ROOT / "data/open_exchange_rates/rub_cis_daily.csv"

        # Act
        rates = baseline.load_rates(path, ["KZT"], cutoff)
        signals = baseline.output_signal_rows(rates, baseline.indicators())

        # Assert
        self.assertEqual(rates["KZT"][-1].day, cutoff)
        self.assertTrue(all(date.fromisoformat(str(row["date"])) <= cutoff for row in signals))


if __name__ == "__main__":
    unittest.main()
