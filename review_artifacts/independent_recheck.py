#!/usr/bin/env python3
"""Independent, dependency-free recheck of the saved KZT V0 OOT events.

This script deliberately does not retrain the branch model.  It freezes the
already emitted OOT policy events and evaluates them against the truth contract
from the latest case Q&A.  The result is therefore a diagnostic audit, not a
replacement for a purged nested walk-forward rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable


HORIZONS = (1, 3, 5, 10, 20)
CLOSING_DELTAS_BPS = (0.0, 10.0, 25.0, 35.0, 50.0, 75.0, 100.0)
NOW_DELTAS_BPS = (10.0, 25.0, 35.0, 50.0, 75.0, 100.0)
PRIMARY_CLOSING_DELTA_BPS = 35.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/kzt_v0/features.csv"))
    parser.add_argument("--backtest", type=Path, default=Path("data/kzt_v0/backtest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("review_artifacts/generated"))
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_903)
    parser.add_argument("--check-known-v0", action="store_true")
    return parser.parse_args()


def load_features(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_backtest(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def row_date(row: dict[str, str]) -> date:
    return date.fromisoformat(row["date"])


def rate(rows: list[dict[str, str]], index: int, key: str = "cbr_rate") -> float:
    return float(rows[index][key])


def oot_indexes(rows: list[dict[str, str]], backtest: dict) -> list[int]:
    result: set[int] = set()
    for fold in backtest["folds"]:
        start = date.fromisoformat(fold["validation_end"])
        end = date.fromisoformat(fold["test_end"])
        result.update(i for i, row in enumerate(rows) if start <= row_date(row) < end)
    return sorted(result)


def fold_end_purged_indexes(
    rows: list[dict[str, str]], backtest: dict, horizon: int
) -> list[int]:
    result: set[int] = set()
    for fold in backtest["folds"]:
        start = date.fromisoformat(fold["validation_end"])
        end = date.fromisoformat(fold["test_end"])
        for i, row in enumerate(rows):
            if not (start <= row_date(row) < end) or i < horizon or i + horizon >= len(rows):
                continue
            if row_date(rows[i + horizon]) < end:
                result.add(i)
    return sorted(result)


def eligible_indexes(rows: list[dict[str, str]], indexes: Iterable[int], horizon: int) -> list[int]:
    # h past observations are required for the official symmetric benefit metric.
    return [i for i in indexes if i >= horizon and i + horizon < len(rows)]


def now_hit(
    rows: list[dict[str, str]],
    index: int,
    horizon: int,
    key: str = "cbr_rate",
    delta_bps: float = 0.0,
) -> int:
    current = rate(rows, index, key)
    future_min = min(rate(rows, j, key) for j in range(index + 1, index + horizon + 1))
    best_wait_gain_bps = max(0.0, (current - future_min) / current * 10_000)
    return int(best_wait_gain_bps <= delta_bps)


def closing_hit(
    rows: list[dict[str, str]], index: int, horizon: int, delta_bps: float, key: str = "cbr_rate"
) -> int:
    current = rate(rows, index, key)
    future_rise_bps = max((rate(rows, j, key) / current - 1.0) * 10_000 for j in range(index + 1, index + horizon + 1))
    # «Курс вырос» требует строгого роста; равенство порогу не считаем hit.
    return int(future_rise_bps > delta_bps)


def symmetric_bps(rows: list[dict[str, str]], index: int, horizon: int, key: str = "cbr_rate") -> float:
    current = rate(rows, index, key)
    window = [rate(rows, j, key) for j in range(index - horizon, index + horizon + 1)]
    return (mean(window) / current - 1.0) * 10_000


def forward_bps(rows: list[dict[str, str]], index: int, horizon: int, key: str = "cbr_rate") -> float:
    current = rate(rows, index, key)
    future = [rate(rows, j, key) for j in range(index + 1, index + horizon + 1)]
    return (mean(future) / current - 1.0) * 10_000


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if not trials:
        return None, None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return centre - half_width, centre + half_width


def fisher_greater(successes_selected: int, selected: int, successes_total: int, total: int) -> float | None:
    if not selected or total <= selected:
        return None
    max_successes = min(selected, successes_total)
    denominator = math.comb(total, selected)
    probability = 0.0
    for successes in range(successes_selected, max_successes + 1):
        failures = selected - successes
        if failures <= total - successes_total:
            favorable_ways = math.comb(successes_total, successes)
            unfavorable_ways = math.comb(total - successes_total, failures)
            probability += favorable_ways * unfavorable_ways / denominator
    return min(1.0, probability)


def moving_block_bootstrap(
    labels: list[int],
    selected: list[int],
    symmetric: list[float],
    forward: list[float],
    replicates: int,
    block_size: int,
    seed: int,
) -> dict[str, list[float | None]]:
    rng = random.Random(seed)
    size = len(labels)
    stats: dict[str, list[float]] = {"lift": [], "delta_pp": [], "symmetric_bps": [], "forward_bps": []}
    if not size:
        return {key: [None, None] for key in stats}
    block_size = min(max(1, block_size), size)
    for _ in range(replicates):
        sample_positions: list[int] = []
        while len(sample_positions) < size:
            start = rng.randrange(size)
            sample_positions.extend((start + offset) % size for offset in range(block_size))
        sample_positions = sample_positions[:size]
        baseline = mean(labels[p] for p in sample_positions)
        signal_positions = [p for p in sample_positions if selected[p]]
        if not signal_positions:
            continue
        hit_rate = mean(labels[p] for p in signal_positions)
        if baseline:
            stats["lift"].append(hit_rate / baseline)
        stats["delta_pp"].append((hit_rate - baseline) * 100)
        stats["symmetric_bps"].append(mean(symmetric[p] for p in signal_positions))
        stats["forward_bps"].append(mean(forward[p] for p in signal_positions))
    return {key: [quantile(values, 0.025), quantile(values, 0.975)] for key, values in stats.items()}


def circular_shift_pvalue(
    values: list[float],
    selected: list[int],
    statistic: Callable[[list[float]], float],
) -> float | None:
    positions = [i for i, flag in enumerate(selected) if flag]
    if not positions or len(values) < 2:
        return None
    observed = statistic([values[i] for i in positions])
    exceedances = 0
    for shift in range(1, len(values)):
        shifted_values = [values[(i + shift) % len(values)] for i in positions]
        exceedances += statistic(shifted_values) >= observed - 1e-12
    return (exceedances + 1) / len(values)


def metric_row(
    rows: list[dict[str, str]],
    universe: list[int],
    event_indexes: set[int],
    scenario: str,
    horizon: int,
    delta_bps: float,
    bootstrap_replicates: int,
    block_size: int,
    seed: int,
) -> dict:
    if scenario == "favorable_now":
        truth = lambda r, i, h: now_hit(r, i, h, delta_bps=delta_bps)
    else:
        truth = lambda r, i, h: closing_hit(r, i, h, delta_bps)
    labels = [truth(rows, i, horizon) for i in universe]
    selected = [int(i in event_indexes) for i in universe]
    signal_positions = [position for position, flag in enumerate(selected) if flag]
    hits = sum(labels[position] for position in signal_positions)
    signals = len(signal_positions)
    baseline = mean(labels) if labels else 0.0
    hit_rate = hits / signals if signals else 0.0
    symmetric = [symmetric_bps(rows, i, horizon) for i in universe]
    forward = [forward_bps(rows, i, horizon) for i in universe]
    selected_symmetric = [symmetric[position] for position in signal_positions]
    selected_forward = [forward[position] for position in signal_positions]
    intervals = moving_block_bootstrap(
        labels, selected, symmetric, forward, bootstrap_replicates, block_size, seed + horizon + int(delta_bps)
    )
    wilson_low, wilson_high = wilson_interval(hits, signals)
    return {
        "scenario": scenario,
        "horizon_moex_rows": horizon,
        "delta_bps": delta_bps,
        "eligible_moex_rows": len(universe),
        "signals": signals,
        "hits": hits,
        "hit_rate": hit_rate,
        "hit_rate_wilson_95": [wilson_low, wilson_high],
        "random_hit_rate": baseline,
        "lift": hit_rate / baseline if baseline else None,
        "lift_block_bootstrap_95": intervals["lift"],
        "absolute_delta_pp": (hit_rate - baseline) * 100,
        "absolute_delta_pp_block_bootstrap_95": intervals["delta_pp"],
        "fisher_exact_greater_p": fisher_greater(hits, signals, sum(labels), len(labels)),
        "symmetric_bps_mean": mean(selected_symmetric) if selected_symmetric else None,
        "symmetric_bps_median": median(selected_symmetric) if selected_symmetric else None,
        "symmetric_bps_block_bootstrap_95": intervals["symmetric_bps"],
        "forward_bps_mean": mean(selected_forward) if selected_forward else None,
        "forward_bps_median": median(selected_forward) if selected_forward else None,
        "forward_bps_block_bootstrap_95": intervals["forward_bps"],
        "circular_shift_p_hit_rate": circular_shift_pvalue(labels, selected, mean),
        "circular_shift_p_symmetric_bps": circular_shift_pvalue(symmetric, selected, mean),
        "circular_shift_p_forward_bps": circular_shift_pvalue(forward, selected, mean),
    }


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


def frequency_metrics(rows: list[dict[str, str]], universe: list[int], events: list[dict]) -> dict:
    start = row_date(rows[universe[0]])
    end = row_date(rows[universe[-1]])
    elapsed_weeks = max((end - start).days / 7, 1 / 7)
    all_weeks: set[date] = set()
    cursor = week_start(start)
    final_week = week_start(end)
    while cursor <= final_week:
        all_weeks.add(cursor)
        cursor += timedelta(days=7)

    result: dict[str, object] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "elapsed_days": (end - start).days,
        "calendar_weeks": len(all_weeks),
    }
    for scenario in ("all", "favorable_now", "window_closing"):
        selected = [e for e in events if scenario == "all" or e["scenario"] == scenario]
        dates = sorted(date.fromisoformat(e["date"]) for e in selected)
        event_weeks = Counter(week_start(day) for day in dates)
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        leading_gap = [max(0, (dates[0] - start).days)] if dates else []
        trailing_gap = [max(0, (end - dates[-1]).days)] if dates else []
        boundary_gaps = leading_gap + gaps + trailing_gap
        month_count = (end.year - start.year) * 12 + end.month - start.month + 1
        active_months = len({(day.year, day.month) for day in dates})
        result[scenario] = {
            "signals": len(dates),
            "signals_per_elapsed_week": len(dates) / elapsed_weeks,
            "silent_week_share": 1 - len(event_weeks) / len(all_weeks),
            "silent_months": month_count - active_months,
            "calendar_months": month_count,
            "median_internal_gap_days": median(gaps) if gaps else None,
            "p90_internal_gap_days": quantile(gaps, 0.9),
            "max_gap_including_boundaries_days": max(boundary_gaps) if boundary_gaps else (end - start).days,
            "cluster_share_internal_gap_le_3d": mean(gap <= 3 for gap in gaps) if gaps else None,
            "weekly_hhi": sum((count / len(dates)) ** 2 for count in event_weeks.values()) if dates else None,
        }
    return result


def source_robustness(
    rows: list[dict[str, str]], universe: list[int], event_indexes: set[int], horizon: int
) -> list[dict]:
    result = []
    for key, label in (("cbr_rate", "CBR"), ("nbk_rate", "NBK"), ("moex_close", "MOEX")):
        labels = [now_hit(rows, i, horizon, key) for i in universe]
        selected_positions = [position for position, index in enumerate(universe) if index in event_indexes]
        hits = sum(labels[position] for position in selected_positions)
        hit_rate = hits / len(selected_positions) if selected_positions else 0.0
        baseline = mean(labels) if labels else 0.0
        result.append(
            {
                "source": label,
                "signals": len(selected_positions),
                "hits": hits,
                "hit_rate": hit_rate,
                "random_hit_rate": baseline,
                "lift": hit_rate / baseline if baseline else None,
            }
        )
    return result


def paired_fast_slow(rows: list[dict[str, str]], events: list[dict]) -> dict:
    index_by_date = {row["date"]: i for i, row in enumerate(rows)}
    pairs = []
    for event in events:
        if event["scenario"] != "window_closing":
            continue
        parent = next(
            (
                item.removeprefix("strong_candidate_on_")
                for item in event.get("evidence", [])
                if item.startswith("strong_candidate_on_")
            ),
            None,
        )
        if parent is None or parent not in index_by_date:
            continue
        fast_index = index_by_date[parent]
        slow_index = int(event["index"])
        fast_rate = rate(rows, fast_index)
        wait_cost = (rate(rows, slow_index) / fast_rate - 1.0) * 10_000
        pairs.append(
            {
                "parent_date": parent,
                "closing_date": event["date"],
                "delay_sessions": slow_index - fast_index,
                "delay_calendar_days": (date.fromisoformat(event["date"]) - date.fromisoformat(parent)).days,
                "wait_cost_bps": wait_cost,
            }
        )
    parents = Counter(pair["parent_date"] for pair in pairs)
    return {
        "pairs": pairs,
        "closing_signals": len(pairs),
        "unique_parent_candidates": len(parents),
        "parents_with_repeated_closing_alerts": sum(count > 1 for count in parents.values()),
        "mean_delay_sessions": mean(pair["delay_sessions"] for pair in pairs) if pairs else None,
        "median_delay_sessions": median(pair["delay_sessions"] for pair in pairs) if pairs else None,
        "mean_wait_cost_bps": mean(pair["wait_cost_bps"] for pair in pairs) if pairs else None,
        "median_wait_cost_bps": median(pair["wait_cost_bps"] for pair in pairs) if pairs else None,
    }


def binomial_survival(successes: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, count) * probability**count * (1 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def optimistic_signal_count_for_power(
    null_rate: float, target_lift: float = 1.3, alpha: float = 0.05, target_power: float = 0.8
) -> dict:
    alternative_rate = min(1.0, null_rate * target_lift)
    for trials in range(2, 5_001):
        critical = next(
            (successes for successes in range(trials + 1) if binomial_survival(successes, trials, null_rate) <= alpha),
            None,
        )
        if critical is None:
            continue
        power = binomial_survival(critical, trials, alternative_rate)
        if power >= target_power:
            return {
                "null_rate": null_rate,
                "alternative_rate": alternative_rate,
                "target_lift": target_lift,
                "alpha_one_sided": alpha,
                "target_power": target_power,
                "required_signals": trials,
                "critical_hits": critical,
                "achieved_power": power,
                "warning": "Optimistic lower bound: known baseline, IID events, one frozen test, no multiplicity.",
            }
    raise RuntimeError("power target was not reached")


def data_quality(rows: list[dict[str, str]]) -> dict:
    zero_range = [float(row["candle_range_bps"]) == 0 for row in rows]
    zero_body = [float(row["candle_body_bps"]) == 0 for row in rows]
    by_period: dict[str, list[bool]] = {"pre_2022": [], "2022": [], "post_2022": []}
    for row, flag in zip(rows, zero_range):
        year = row_date(row).year
        period = "pre_2022" if year < 2022 else "2022" if year == 2022 else "post_2022"
        by_period[period].append(flag)
    return {
        "rows": len(rows),
        "start": rows[0]["date"],
        "end": rows[-1]["date"],
        "cbr_fresh_share": mean(float(row["cbr_is_fresh"]) for row in rows),
        "nbk_fresh_share": mean(float(row["nbk_is_fresh"]) for row in rows),
        "moex_zero_range_share": mean(zero_range),
        "moex_zero_body_share": mean(zero_body),
        "moex_zero_range_share_by_period": {
            key: mean(values) if values else None for key, values in by_period.items()
        },
        "max_cbr_age_days": max(float(row["cbr_age_days"]) for row in rows),
        "max_nbk_age_days": max(float(row["nbk_age_days"]) for row in rows),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in row.items()
                }
            )


def format_ci(value: list[float | None]) -> str:
    if value[0] is None:
        return "n/a"
    return f"[{value[0]:.3f}; {value[1]:.3f}]"


def write_markdown(summary: dict, path: Path) -> None:
    primary = [
        row
        for row in summary["corrected_metrics"]
        if row["horizon_moex_rows"] == 5
        and (row["scenario"] == "favorable_now" or row["delta_bps"] == PRIMARY_CLOSING_DELTA_BPS)
    ]
    lines = [
        "# Независимая перепроверка сохранённых KZT V0 событий",
        "",
        "Это fixed-event диагностика: даты OOT-сигналов заморожены, truth пересчитан по Q&A. "
        "Она не исправляет leakage и не заменяет полный nested walk-forward rerun.",
        "`h_moex_rows` считает следующие строки сохранённой MOEX-aligned панели; это не distinct "
        "CBR publications и не календарные дни.",
        "",
        "| Scenario | h_moex_rows | δ, bps | Hits/signals | Random | Lift | "
        "Lift block-bootstrap 95% | ±h bps | Forward bps | Fisher p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['scenario']} | {row['horizon_moex_rows']} | {row['delta_bps']:.0f} | "
            f"{row['hits']}/{row['signals']} | {row['random_hit_rate']:.3f} | {row['lift']:.3f} | "
            f"{format_ci(row['lift_block_bootstrap_95'])} | {row['symmetric_bps_mean']:.1f} | "
            f"{row['forward_bps_mean']:.1f} | {row['fisher_exact_greater_p']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity: полностью разрешённый outcome внутри каждого fold",
            "",
            "Последние h_moex_rows строк каждого test fold исключены, поэтому выборка ещё меньше.",
            "",
            "| Scenario | h_moex_rows | δ, bps | Hits/signals | Random | Lift | ±h bps | Forward bps |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["strict_fold_end_purged_h5_moex_rows"]:
        lines.append(
            f"| {row['scenario']} | {row['horizon_moex_rows']} | {row['delta_bps']:.0f} | "
            f"{row['hits']}/{row['signals']} | {row['random_hit_rate']:.3f} | {row['lift']:.3f} | "
            f"{row['symmetric_bps_mean']:.1f} | {row['forward_bps_mean']:.1f} |"
        )
    frequency = summary["frequency"]
    lines.extend(
        [
            "",
            "## Частота на полном OOT-span",
            "",
            f"Период: {frequency['start']}…{frequency['end']} "
            f"({frequency['calendar_weeks']} календарных недель).",
            "",
            "| Scenario | Signals | Signals/week | Silent weeks | Max gap incl. boundaries |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in ("all", "favorable_now", "window_closing"):
        item = frequency[scenario]
        lines.append(
            f"| {scenario} | {item['signals']} | {item['signals_per_elapsed_week']:.3f} | "
            f"{item['silent_week_share']:.1%} | {item['max_gap_including_boundaries_days']} d |"
        )
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- `h_moex_rows` — следующие строки MOEX-aligned feature panel, не distinct CBR publications "
            "и не календарные дни; CBR значения внутри окна могут быть carry-forward.",
            "- Ветки `favorable_now` и `window_closing` нельзя сравнивать с "
            "опубликованными headline-числами: там другие/некорректные truth-функции.",
            "- Интервалы — диагностический circular moving-block bootstrap по 20 MOEX-aligned rows; "
            "при 7/10 сигналах они неизбежно нестабильны.",
            "- Fisher p и circular-shift p не скорректированы за перебор моделей, "
            "порогов, горизонтов и сценариев.",
            "- `required_signals` ниже — оптимистичная нижняя граница мощности, "
            "а не план эксперимента.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_features(args.features)
    backtest = load_backtest(args.backtest)
    oot = oot_indexes(rows, backtest)
    events = [event for event in backtest["events"] if event.get("policy_eligible")]
    events_by_scenario = {
        scenario: {int(event["index"]) for event in events if event["scenario"] == scenario}
        for scenario in ("favorable_now", "window_closing")
    }

    metrics = []
    sensitivity_metrics = []
    for horizon in HORIZONS:
        universe = eligible_indexes(rows, oot, horizon)
        metrics.append(
            metric_row(
                rows,
                universe,
                events_by_scenario["favorable_now"],
                "favorable_now",
                horizon,
                0.0,
                args.bootstrap_replicates,
                args.block_size,
                args.seed,
            )
        )
        metrics.append(
            metric_row(
                rows,
                universe,
                events_by_scenario["window_closing"],
                "window_closing",
                horizon,
                PRIMARY_CLOSING_DELTA_BPS,
                args.bootstrap_replicates,
                args.block_size,
                args.seed,
            )
        )
        for delta_bps in NOW_DELTAS_BPS:
            sensitivity_metrics.append(
                metric_row(
                    rows,
                    universe,
                    events_by_scenario["favorable_now"],
                    "favorable_now",
                    horizon,
                    delta_bps,
                    0,
                    args.block_size,
                    args.seed,
                )
            )
        for delta_bps in CLOSING_DELTAS_BPS:
            if delta_bps == PRIMARY_CLOSING_DELTA_BPS:
                continue
            sensitivity_metrics.append(
                metric_row(
                    rows,
                    universe,
                    events_by_scenario["window_closing"],
                    "window_closing",
                    horizon,
                    delta_bps,
                    0,
                    args.block_size,
                    args.seed,
                )
            )

    strict_h5_universe = fold_end_purged_indexes(rows, backtest, 5)
    strict_h5 = [
        metric_row(
            rows,
            strict_h5_universe,
            events_by_scenario[scenario],
            scenario,
            5,
            0.0 if scenario == "favorable_now" else PRIMARY_CLOSING_DELTA_BPS,
            args.bootstrap_replicates,
            args.block_size,
            args.seed,
        )
        for scenario in ("favorable_now", "window_closing")
    ]

    h5_universe = eligible_indexes(rows, oot, 5)
    primary_now = next(
        row
        for row in metrics
        if row["scenario"] == "favorable_now" and row["horizon_moex_rows"] == 5
    )
    primary_closing = next(
        row
        for row in metrics
        if row["scenario"] == "window_closing"
        and row["horizon_moex_rows"] == 5
        and row["delta_bps"] == PRIMARY_CLOSING_DELTA_BPS
    )
    summary = {
        "scope": {
            "kind": "fixed saved OOT event diagnostic",
            "target": "CBR RUB per KZT",
            "horizons_moex_rows": HORIZONS,
            "decision_clock": (
                "successive rows of the MOEX-aligned saved feature panel; not distinct CBR publications or "
                "calendar days; CBR values can be carried forward"
            ),
            "primary_closing_delta_bps": PRIMARY_CLOSING_DELTA_BPS,
            "bootstrap_replicates": args.bootstrap_replicates,
            "block_size_moex_rows": args.block_size,
            "seed": args.seed,
            "warnings": [
                "Saved events inherit model-selection and validation-boundary defects from edelkin_test.",
                "This recheck does not establish deployable out-of-time performance.",
                "Every horizon is counted in MOEX-aligned panel rows, not CBR publication sessions.",
                "All uncertainty calculations are exploratory and unadjusted "
                "for model/horizon/threshold multiplicity.",
            ],
        },
        "corrected_metrics": metrics,
        "sensitivity_metrics": sensitivity_metrics,
        "strict_fold_end_purged_h5_moex_rows": strict_h5,
        "frequency": frequency_metrics(rows, h5_universe, events),
        "source_robustness_now_h5_moex_rows": source_robustness(
            rows, h5_universe, events_by_scenario["favorable_now"], 5
        ),
        "paired_fast_slow": paired_fast_slow(rows, events),
        "data_quality": data_quality(rows),
        "optimistic_power_lower_bounds": {
            "now_h5_moex_rows_lift_1_3": optimistic_signal_count_for_power(primary_now["random_hit_rate"]),
            "closing_h5_moex_rows_delta35_lift_1_3": optimistic_signal_count_for_power(
                primary_closing["random_hit_rate"]
            ),
        },
    }

    if args.check_known_v0:
        assert len(rows) == 1_295, len(rows)
        assert len(oot) == 443, len(oot)
        assert len(events) == 17, len(events)
        assert primary_now["signals"] == 7 and primary_now["hits"] == 4, primary_now
        assert primary_closing["signals"] == 10 and primary_closing["hits"] == 9, primary_closing

    rounded_summary = rounded(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(rounded_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv([rounded(row) for row in metrics], args.output_dir / "corrected_event_metrics.csv")
    write_csv([rounded(row) for row in sensitivity_metrics], args.output_dir / "delta_sensitivity_metrics.csv")
    write_markdown(rounded_summary, args.output_dir / "recheck_summary.md")
    print(json.dumps({"output_dir": str(args.output_dir), "checks": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
