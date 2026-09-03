#!/usr/bin/env python3
"""Audit FX history and evaluate transparent hackathon signal baselines.

The script intentionally uses only the Python standard library.  It is not the
final ML model: it creates a trustworthy floor, catches data/leakage problems
and produces the exact matrices needed to decide which ideas deserve modelling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence


CORE_QUOTES = ("AMD", "KGS", "KZT", "TJS", "UZS")
HORIZONS = (1, 3, 5, 10, 20)
EPSILON = 1e-12


@dataclass(frozen=True)
class RatePoint:
    day: date
    rate: float
    published_at: datetime


@dataclass(frozen=True)
class Candidate:
    index: int
    strength: float


@dataclass(frozen=True)
class Indicator:
    name: str
    scenario: str
    direction: str
    speed_days: int
    warmup_days: int
    cooldown_days: int
    detector: Callable[[Sequence[RatePoint]], list[Candidate]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit daily RUB/CIS rates and run leakage-safe rule baselines."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/open_exchange_rates/rub_cis_daily.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Researches/ml_hackathon/results")
    )
    parser.add_argument("--quotes", nargs="+", default=list(CORE_QUOTES))
    parser.add_argument("--as-of", type=date.fromisoformat)
    return parser.parse_args()


def load_rates(path: Path, quotes: Iterable[str], as_of: date | None = None) -> dict[str, list[RatePoint]]:
    selected = set(quotes)
    rates: dict[str, list[RatePoint]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"date", "quote", "rub_per_quote", "published_at_utc"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            quote = row["quote"]
            day = date.fromisoformat(row["date"])
            if quote not in selected or (as_of is not None and day > as_of):
                continue
            point = RatePoint(
                day=day,
                rate=float(row["rub_per_quote"]),
                published_at=datetime.fromisoformat(row["published_at_utc"]),
            )
            rates[quote].append(point)

    absent = selected - set(rates)
    if absent:
        raise ValueError(f"No data for quotes: {', '.join(sorted(absent))}")
    for quote in rates:
        rates[quote].sort(key=lambda point: point.day)
    return dict(rates)


def is_same_rate(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=EPSILON, abs_tol=EPSILON)


def is_updated(points: Sequence[RatePoint], index: int) -> bool:
    return index > 0 and not is_same_rate(points[index].rate, points[index - 1].rate)


def percentile_rank(values: Sequence[float], value: float) -> float:
    below = sum(candidate < value for candidate in values)
    equal = sum(is_same_rate(candidate, value) for candidate in values)
    return (below + 0.5 * equal) / len(values)


def changed_indices(points: Sequence[RatePoint]) -> list[int]:
    return [index for index in range(1, len(points)) if is_updated(points, index)]


def detect_momentum_down_3(points: Sequence[RatePoint]) -> list[Candidate]:
    result: list[Candidate] = []
    updated = changed_indices(points)
    for position in range(2, len(updated)):
        indices = updated[position - 2 : position + 1]
        changes = [
            (points[index].rate - points[index - 1].rate) / points[index - 1].rate
            for index in indices
        ]
        if all(change < 0 for change in changes):
            strength = min(100.0, abs(sum(changes)) * 10_000.0)
            result.append(Candidate(indices[-1], strength))
    return result


def detect_low_level_90_p10(points: Sequence[RatePoint]) -> list[Candidate]:
    result: list[Candidate] = []
    for index in range(89, len(points)):
        if not is_updated(points, index):
            continue
        window = [point.rate for point in points[index - 89 : index + 1]]
        rank = percentile_rank(window, points[index].rate)
        if rank <= 0.10:
            strength = min(100.0, max(0.0, (0.10 - rank) / 0.10 * 100.0))
            result.append(Candidate(index, strength))
    return result


def detect_reversal_up_from_30d_low(points: Sequence[RatePoint]) -> list[Candidate]:
    result: list[Candidate] = []
    updated = changed_indices(points)
    for position in range(1, len(updated)):
        index = updated[position]
        previous = updated[position - 1]
        if index < 30 or points[index].rate <= points[previous].rate:
            continue
        prior_window = [point.rate for point in points[index - 30 : index]]
        low = min(prior_window)
        distance_from_low_bps = (points[previous].rate - low) / points[previous].rate * 10_000.0
        if distance_from_low_bps <= 25.0:
            rise_bps = (points[index].rate - points[previous].rate) / points[previous].rate * 10_000.0
            result.append(Candidate(index, min(100.0, max(0.0, rise_bps))))
    return result


def detect_seasonal_positive_month(points: Sequence[RatePoint]) -> list[Candidate]:
    """Past-only expanding calendar baseline, evaluated at month level."""
    history: dict[int, list[float]] = defaultdict(list)
    result: list[Candidate] = []
    previous_month: tuple[int, int] | None = None
    for index in range(1, len(points)):
        point = points[index]
        month_key = (point.day.year, point.day.month)
        if is_updated(points, index):
            samples = history[point.day.month]
            if month_key != previous_month and len(samples) >= 60:
                mean_change_bps = statistics.fmean(samples) * 10_000.0
                if mean_change_bps > 0:
                    result.append(Candidate(index, min(100.0, mean_change_bps * 10.0)))
            change = (point.rate - points[index - 1].rate) / points[index - 1].rate
            history[point.day.month].append(change)
            previous_month = month_key
    return result


def indicators() -> tuple[Indicator, ...]:
    return (
        Indicator(
            name="momentum_down_3",
            scenario="NOW_FAVORABLE",
            direction="recipient_currency_cheaper_in_rub",
            speed_days=3,
            warmup_days=5,
            cooldown_days=4,
            detector=detect_momentum_down_3,
        ),
        Indicator(
            name="low_level_90_p10",
            scenario="NOW_FAVORABLE",
            direction="historically_low_rub_per_quote",
            speed_days=90,
            warmup_days=90,
            cooldown_days=4,
            detector=detect_low_level_90_p10,
        ),
        Indicator(
            name="reversal_up_from_30d_low",
            scenario="WINDOW_CLOSING",
            direction="recipient_currency_started_rising_in_rub",
            speed_days=1,
            warmup_days=30,
            cooldown_days=4,
            detector=detect_reversal_up_from_30d_low,
        ),
        Indicator(
            name="seasonal_positive_month",
            scenario="WINDOW_CLOSING",
            direction="historically_rising_calendar_month",
            speed_days=365,
            warmup_days=730,
            cooldown_days=4,
            detector=detect_seasonal_positive_month,
        ),
    )


def apply_cooldown(
    points: Sequence[RatePoint], candidates: Sequence[Candidate], cooldown_days: int
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
            continue
        gap = (points[candidate.index].day - points[selected[-1].index].day).days
        if gap >= cooldown_days:
            selected.append(candidate)
    return selected


def strict_hit(points: Sequence[RatePoint], index: int, horizon: int, scenario: str) -> bool:
    current = points[index].rate
    future = [point.rate for point in points[index + 1 : index + horizon + 1]]
    if scenario == "NOW_FAVORABLE":
        return current <= min(future) + EPSILON
    if scenario == "WINDOW_CLOSING":
        return points[index + horizon].rate > current + EPSILON
    raise ValueError(f"Unknown scenario: {scenario}")


def symmetric_benefit_bps(points: Sequence[RatePoint], index: int, horizon: int) -> float:
    window = [point.rate for point in points[index - horizon : index + horizon + 1]]
    current = points[index].rate
    return (statistics.fmean(window) - current) / current * 10_000.0


def forward_benefit_bps(points: Sequence[RatePoint], index: int, horizon: int) -> float:
    future = [point.rate for point in points[index + 1 : index + horizon + 1]]
    current = points[index].rate
    return (statistics.fmean(future) - current) / current * 10_000.0


def month_block_interval(
    dated_values: Sequence[tuple[date, float]], seed: int, repetitions: int = 1_000
) -> tuple[float, float]:
    if not dated_values:
        return math.nan, math.nan
    blocks: dict[tuple[int, int], list[float]] = defaultdict(list)
    for day, value in dated_values:
        blocks[(day.year, day.month)].append(value)
    block_values = list(blocks.values())
    if len(block_values) < 2:
        value = statistics.fmean(value for _, value in dated_values)
        return value, value
    generator = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = [generator.choice(block_values) for _ in block_values]
        estimates.append(statistics.fmean(value for block in sampled for value in block))
    estimates.sort()
    return estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions) - 1]


def gap_statistics(points: Sequence[RatePoint], candidates: Sequence[Candidate]) -> tuple[float, float]:
    if len(candidates) < 2:
        return 0.0, math.nan
    gaps = [
        (points[right.index].day - points[left.index].day).days
        for left, right in zip(candidates, candidates[1:])
    ]
    clustered_share = sum(gap <= 3 for gap in gaps) / len(gaps)
    mean_gap = statistics.fmean(gaps)
    gap_cv = statistics.pstdev(gaps) / mean_gap if mean_gap else math.nan
    return clustered_share, gap_cv


def eligible_indices(
    points: Sequence[RatePoint], horizon: int, period: str, warmup_days: int
) -> list[int]:
    result = []
    for index in range(max(horizon, warmup_days), len(points) - horizon):
        if not is_updated(points, index):
            continue
        if period != "ALL" and points[index].day.year != int(period):
            continue
        result.append(index)
    return result


def evaluate_period(
    quote: str,
    points: Sequence[RatePoint],
    indicator: Indicator,
    raw: Sequence[Candidate],
    selected: Sequence[Candidate],
    horizon: int,
    period: str,
) -> dict[str, object] | None:
    eligible = set(eligible_indices(points, horizon, period, indicator.warmup_days))
    signals = [candidate for candidate in selected if candidate.index in eligible]
    raw_signals = [candidate for candidate in raw if candidate.index in eligible]
    if not eligible:
        return None

    baseline_hits = [strict_hit(points, index, horizon, indicator.scenario) for index in eligible]
    baseline_hit_rate = statistics.fmean(baseline_hits)
    span = max(
        1,
        (max(points[index].day for index in eligible) - min(points[index].day for index in eligible)).days + 1,
    )
    raw_cluster, _ = gap_statistics(points, raw_signals)
    if not signals:
        return {
            "period": period,
            "corridor": f"RUB/{quote}",
            "indicator": indicator.name,
            "scenario": indicator.scenario,
            "horizon_days": horizon,
            "raw_candidate_count": len(raw_signals),
            "signal_count": 0,
            "hit_rate": math.nan,
            "random_day_hit_rate": baseline_hit_rate,
            "lift": math.nan,
            "benefit_bps_symmetric": math.nan,
            "benefit_bps_ci_low": math.nan,
            "benefit_bps_ci_high": math.nan,
            "benefit_bps_forward": math.nan,
            "signals_per_week": 0.0,
            "raw_cluster_share_gap_le_3d": raw_cluster,
            "post_cooldown_cluster_share_gap_le_3d": 0.0,
            "post_cooldown_gap_cv": math.nan,
        }

    signal_hits = [strict_hit(points, candidate.index, horizon, indicator.scenario) for candidate in signals]
    hit_rate = statistics.fmean(signal_hits)
    lift = hit_rate / baseline_hit_rate if baseline_hit_rate else math.nan
    symmetric = [
        (points[candidate.index].day, symmetric_benefit_bps(points, candidate.index, horizon))
        for candidate in signals
    ]
    forward = [forward_benefit_bps(points, candidate.index, horizon) for candidate in signals]
    interval = month_block_interval(symmetric, seed=sum(ord(char) for char in f"{quote}{indicator.name}{horizon}{period}"))
    selected_cluster, gap_cv = gap_statistics(points, signals)
    signals_per_week = len(signals) / span * 7.0
    return {
        "period": period,
        "corridor": f"RUB/{quote}",
        "indicator": indicator.name,
        "scenario": indicator.scenario,
        "horizon_days": horizon,
        "raw_candidate_count": len(raw_signals),
        "signal_count": len(signals),
        "hit_rate": hit_rate,
        "random_day_hit_rate": baseline_hit_rate,
        "lift": lift,
        "benefit_bps_symmetric": statistics.fmean(value for _, value in symmetric),
        "benefit_bps_ci_low": interval[0],
        "benefit_bps_ci_high": interval[1],
        "benefit_bps_forward": statistics.fmean(forward),
        "signals_per_week": signals_per_week,
        "raw_cluster_share_gap_le_3d": raw_cluster,
        "post_cooldown_cluster_share_gap_le_3d": selected_cluster,
        "post_cooldown_gap_cv": gap_cv,
    }


def audit_rates(rates: dict[str, list[RatePoint]]) -> dict[str, object]:
    corridors = {}
    for quote, points in sorted(rates.items()):
        duplicate_days = len(points) - len({point.day for point in points})
        gaps = [(right.day - left.day).days for left, right in zip(points, points[1:])]
        unchanged = [index for index in range(1, len(points)) if not is_updated(points, index)]
        publication_after_day = sum(point.published_at.date() > point.day for point in points)
        corridors[f"RUB/{quote}"] = {
            "rows": len(points),
            "first_date": points[0].day.isoformat(),
            "last_date": points[-1].day.isoformat(),
            "duplicate_dates": duplicate_days,
            "non_daily_gaps": sum(gap != 1 for gap in gaps),
            "non_positive_rates": sum(point.rate <= 0 for point in points),
            "unchanged_days": len(unchanged),
            "unchanged_share": len(unchanged) / max(1, len(points) - 1),
            "unchanged_weekend_share": (
                sum(points[index].day.weekday() >= 5 for index in unchanged) / len(unchanged)
                if unchanged
                else 0.0
            ),
            "publication_date_after_rate_date": publication_after_day,
        }
    return {
        "rate_field": "rub_per_quote",
        "rate_orientation": "lower_is_better_for_sender",
        "signal_time_contract": "point is usable only at or after published_at_utc",
        "corridors": corridors,
    }


def output_signal_rows(
    rates: dict[str, list[RatePoint]], specs: Sequence[Indicator]
) -> list[dict[str, object]]:
    rows = []
    for quote, points in sorted(rates.items()):
        for spec in specs:
            selected = apply_cooldown(points, spec.detector(points), spec.cooldown_days)
            for candidate in selected:
                point = points[candidate.index]
                speed_class = "fast" if spec.speed_days <= 3 else "medium" if spec.speed_days <= 20 else "slow"
                rows.append(
                    {
                        "date": point.day.isoformat(),
                        "corridor": f"RUB/{quote}",
                        "indicator": spec.name,
                        "direction": spec.direction,
                        "strength_0_100": round(candidate.strength, 4),
                        "speed_days": spec.speed_days,
                        "speed_class": speed_class,
                        "recommended_scenario": spec.scenario,
                        "rate": point.rate,
                        "rate_field": "rub_per_quote",
                        "as_of_utc": point.published_at.isoformat(),
                    }
                )
    return sorted(rows, key=lambda row: (str(row["date"]), str(row["corridor"]), str(row["indicator"])))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_verdicts(metrics: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    overall = [row for row in metrics if row["period"] == "ALL"]
    yearly = [row for row in metrics if row["period"] != "ALL"]
    verdicts = []
    for row in overall:
        related = [
            candidate
            for candidate in yearly
            if candidate["corridor"] == row["corridor"]
            and candidate["indicator"] == row["indicator"]
            and candidate["horizon_days"] == row["horizon_days"]
        ]
        positive_oot = sum(float(candidate["lift"]) >= 1.0 for candidate in related)
        target_oot = sum(float(candidate["lift"]) >= 1.3 for candidate in related)
        frequency_ok = 1.0 <= float(row["signals_per_week"]) <= 2.0
        statistically_positive = float(row["benefit_bps_ci_low"]) > 0
        if (
            float(row["lift"]) >= 1.3
            and statistically_positive
            and frequency_ok
            and related
            and positive_oot / len(related) >= 0.6
        ):
            verdict = "KEEP_FOR_ML"
        elif float(row["lift"]) < 1.0 or float(row["benefit_bps_symmetric"]) <= 0:
            verdict = "DROP_OR_NEGATIVE_CONTROL"
        else:
            verdict = "RESEARCH_ONLY"
        verdicts.append(
            {
                **row,
                "oot_periods": len(related),
                "oot_lift_ge_1_count": positive_oot,
                "oot_lift_ge_1_3_count": target_oot,
                "frequency_gate": frequency_ok,
                "benefit_ci_positive_gate": statistically_positive,
                "baseline_verdict": verdict,
            }
        )
    return verdicts


def run(rates: dict[str, list[RatePoint]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs = indicators()
    metrics: list[dict[str, object]] = []
    for quote, points in sorted(rates.items()):
        years = sorted({point.day.year for point in points if point.day.year >= 2020})
        for spec in specs:
            raw = spec.detector(points)
            selected = apply_cooldown(points, raw, spec.cooldown_days)
            for horizon in HORIZONS:
                for period in ["ALL", *[str(year) for year in years]]:
                    row = evaluate_period(quote, points, spec, raw, selected, horizon, period)
                    if row is not None:
                        metrics.append(row)
    return metrics, output_signal_rows(rates, specs)


def main() -> int:
    args = parse_args()
    rates = load_rates(args.input, args.quotes, args.as_of)
    metrics, signals = run(rates)
    verdicts = build_verdicts(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "data_audit.json").open("w", encoding="utf-8") as target:
        json.dump(audit_rates(rates), target, ensure_ascii=False, indent=2)
        target.write("\n")
    write_csv(args.output_dir / "baseline_metrics.csv", metrics)
    write_csv(args.output_dir / "baseline_verdicts.csv", verdicts)
    write_csv(args.output_dir / "signals.csv", signals)
    manifest = {
        "input": str(args.input),
        "as_of": args.as_of.isoformat() if args.as_of else None,
        "quotes": args.quotes,
        "horizons": list(HORIZONS),
        "indicators": [asdict(spec) | {"detector": spec.detector.__name__} for spec in indicators()],
        "outputs": ["data_audit.json", "baseline_metrics.csv", "baseline_verdicts.csv", "signals.csv"],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(f"Audited {sum(len(points) for points in rates.values())} rows across {len(rates)} corridors")
    print(f"Wrote {len(metrics)} metric rows and {len(signals)} signals to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
