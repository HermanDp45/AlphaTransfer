"""Walk-forward evaluation. Future prices are used only after a signal exists."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from statistics import mean, pstdev
from typing import Iterable, Sequence

from .cbr import Quote
from .engine import SignalConfig, indicator_rows, signals_as_of
from .ml import LocalMinimumLogit, ml_events_for_rows


def _by_corridor(quotes: Iterable[Quote]) -> dict[str, list[Quote]]:
    grouped: dict[str, list[Quote]] = {}
    for q in quotes:
        grouped.setdefault(q.corridor, []).append(q)
    return {k: sorted(v, key=lambda q: q.date) for k, v in grouped.items()}


def _outcome(series: Sequence[Quote], idx: int, horizon: int, scenario: str, rise_threshold_bps: float) -> tuple[bool, float] | None:
    if idx < horizon or idx + horizon >= len(series):
        return None
    now = series[idx].rub_per_unit
    future = [q.rub_per_unit for q in series[idx + 1 : idx + horizon + 1]]
    local = [q.rub_per_unit for q in series[idx - horizon : idx + horizon + 1]]
    # Positive bps always means a client-beneficial value of the signal date.
    if scenario == "favourable_now":
        hit = now <= min(future)
        benefit_bps = (mean(local) / now - 1) * 10_000
    elif scenario == "window_closing":
        change_bps = (future[-1] / now - 1) * 10_000
        hit = change_bps >= rise_threshold_bps
        # Keep the business-value metric comparable across both message
        # types: price at signal versus mean of the symmetric ±h window.
        benefit_bps = (mean(local) / now - 1) * 10_000
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return hit, benefit_bps


def evaluate_events(
    quotes: Iterable[Quote], events: Iterable[dict], *, horizon: int = 5, rise_threshold_bps: float = 0.0,
    evaluation_start: date | None = None, evaluation_end: date | None = None,
) -> dict:
    """Metrics by indicator/corridor/scenario, including random-day baseline."""
    by_corridor = _by_corridor(quotes)
    indexed = {c: {q.date: i for i, q in enumerate(series)} for c, series in by_corridor.items()}
    buckets: dict[tuple[str, str, str], list[tuple[bool, float, date]]] = {}
    for event in events:
        corridor, scenario = event["corridor"], event["scenario"]
        if (evaluation_start is not None and event["date"] < evaluation_start) or (evaluation_end is not None and event["date"] > evaluation_end):
            continue
        if corridor not in indexed or event["date"] not in indexed[corridor]:
            continue
        event_index = indexed[corridor][event["date"]]
        # An as-of report may not use a later price even merely to settle its
        # most recent messages; those messages remain pending until tomorrow.
        if evaluation_end is not None and (event_index + horizon >= len(by_corridor[corridor]) or by_corridor[corridor][event_index + horizon].date > evaluation_end):
            continue
        outcome = _outcome(by_corridor[corridor], event_index, horizon, scenario, rise_threshold_bps)
        if outcome is not None:
            buckets.setdefault((event["indicator"], corridor, scenario), []).append((*outcome, event["date"]))
    result: list[dict] = []
    for (indicator, corridor, scenario), resolved in sorted(buckets.items()):
        series = by_corridor[corridor]
        baseline = [
            o for i, quote in enumerate(series)
            if (evaluation_start is None or quote.date >= evaluation_start)
            and (evaluation_end is None or quote.date <= evaluation_end)
            # Do not settle either signal or baseline using prices after the
            # declared evaluation cutoff.
            and (evaluation_end is None or (i + horizon < len(series) and series[i + horizon].date <= evaluation_end))
            and (o := _outcome(series, i, horizon, scenario, rise_threshold_bps)) is not None
        ]
        if not baseline:
            continue
        hit_rate = mean(x[0] for x in resolved)
        random_rate = mean(x[0] for x in baseline)
        dates = sorted(x[2] for x in resolved)
        span_days = max((dates[-1] - dates[0]).days, 1)
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        week_counts: dict[tuple[int, int], int] = {}
        for d in dates:
            iso = d.isocalendar()
            week_counts[(iso.year, iso.week)] = week_counts.get((iso.year, iso.week), 0) + 1
        result.append({
            "indicator": indicator, "corridor": corridor, "scenario": scenario, "horizon": horizon,
            "signals": len(resolved), "hit_rate": round(hit_rate, 4), "random_day_hit_rate": round(random_rate, 4),
            "lift": round(hit_rate / random_rate, 4) if random_rate else None,
            "mean_benefit_bps": round(mean(x[1] for x in resolved), 2),
            "random_mean_benefit_bps": round(mean(x[1] for x in baseline), 2),
            "signals_per_week": round(len(resolved) / (span_days / 7), 3),
            "signals_per_month": round(len(resolved) / (span_days / 30.44), 3),
            "cluster_share_gap_le_3d": round(mean(g <= 3 for g in gaps), 4) if gaps else 0.0,
            "interval_std_days": round(pstdev(gaps), 3) if len(gaps) > 1 else 0.0,
            "max_signals_in_week": max(week_counts.values()),
        })
    return {"metrics": result, "horizon": horizon, "evaluation_start": evaluation_start.isoformat() if evaluation_start else None, "evaluation_end": evaluation_end.isoformat() if evaluation_end else None, "definition": "favourable_now: signal price remains <= every next h publication-day price; window_closing: price rises by threshold over h days"}


def _score(report: dict) -> float:
    """Asymmetric utility: false positive costs 3x a missed valid day."""
    rows = report["metrics"]
    if not rows:
        return float("-inf")
    utilities = []
    for row in rows:
        hit, rate = row["hit_rate"], row["signals_per_week"]
        # Target 1--2 / week; do not choose an accurate but unusable flood.
        frequency_penalty = max(0.0, rate - 2.0) * 0.20 + max(0.0, 1.0 - rate) * 0.05
        utilities.append(4 * hit - 3 - frequency_penalty)
    return mean(utilities)


def walk_forward_backtest(
    quotes: Iterable[Quote], candidates: Sequence[SignalConfig], *, train_observations: int = 252, test_observations: int = 63, horizon: int = 5
) -> dict:
    """Select config on past resolved labels and report independent test folds.

    A fold's test signals are generated with their date as ``as_of``; future
    data is not supplied to the signal engine. The later prices enter only
    ``evaluate_events`` to settle the already-issued messages.
    """
    all_quotes = list(quotes)
    if not candidates:
        raise ValueError("At least one SignalConfig candidate is required")
    grouped = _by_corridor(all_quotes)
    folds: list[dict] = []
    for corridor, series in grouped.items():
        start = train_observations + horizon
        while start + test_observations + horizon <= len(series):
            train_end = series[start - 1].date
            test_end = series[start + test_observations - 1].date
            train_quotes = [q for q in series[:start] if q.date <= train_end]
            choices: list[tuple[float, SignalConfig]] = []
            for config in candidates:
                train_events = signals_as_of(train_quotes, train_end, config)
                train_report = evaluate_events(train_quotes, train_events, horizon=horizon, evaluation_end=train_end)
                choices.append((_score(train_report), config))
            score, chosen = max(choices, key=lambda pair: pair[0])
            # Passing the full series is safe because signals_as_of ignores
            # later rows by date. It mirrors the exact live API contract.
            test_events = [e for e in signals_as_of(series, test_end, chosen) if train_end < e["date"] <= test_end]
            resolved_events = [e for e in test_events if e["date"] <= series[start + test_observations - horizon - 1].date]
            test_report = evaluate_events(series, resolved_events, horizon=horizon, evaluation_start=series[start].date, evaluation_end=series[start + test_observations - 1].date)
            folds.append({"corridor": corridor, "train_end": train_end.isoformat(), "test_end": test_end.isoformat(), "chosen_config": asdict(chosen), "training_utility": round(score, 4), "test": test_report})
            start += test_observations
    all_metrics = [row for fold in folds for row in fold["test"]["metrics"]]
    return {"method": "expanding walk-forward; select on prior resolved observations; evaluate next block", "folds": folds, "out_of_time_metrics": all_metrics}


def walk_forward_ml_backtest(
    quotes: Iterable[Quote], config: SignalConfig = SignalConfig(), *, train_observations: int = 252,
    test_observations: int = 63, horizon: int = 5, false_positive_cost: float = 3.0,
) -> dict:
    """Out-of-time evaluation for classical ML, frozen once per test fold."""
    folds: list[dict] = []
    for corridor, series in _by_corridor(quotes).items():
        start = train_observations + horizon
        while start + test_observations + horizon <= len(series):
            train_end, test_end = series[start - 1].date, series[start + test_observations - 1].date
            model = LocalMinimumLogit(horizon=horizon, false_positive_cost=false_positive_cost).fit(series[:start], train_end, config)
            # All feature windows end on their own row; evaluating features
            # through test_end therefore does not reveal a later price.  We
            # calculate them once per fold instead of once per test date.
            latest_resolved = series[start + test_observations - horizon - 1].date
            rows = [
                row for row in indicator_rows(series, config, as_of=test_end)
                if series[start].date <= row["date"] <= latest_resolved
            ]
            events = ml_events_for_rows(rows, model)
            report = evaluate_events(series, events, horizon=horizon, evaluation_start=series[start].date, evaluation_end=test_end)
            folds.append({"corridor": corridor, "train_end": train_end.isoformat(), "test_end": test_end.isoformat(), "model": model.metadata(), "test": report})
            start += test_observations
    return {"method": "walk-forward logistic regression, frozen per test fold", "folds": folds, "out_of_time_metrics": [m for f in folds for m in f["test"]["metrics"]]}
