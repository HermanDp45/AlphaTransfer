"""Expanding walk-forward selection and out-of-time metrics."""

from __future__ import annotations

from datetime import date
from statistics import mean, pstdev
from typing import Callable

from .features import feature_names, local_minimum_label, window_closing_label
from .models import LogisticModel, ShallowBoostingModel
from .policy import apply_policy, raw_candidates


def add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    import calendar
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _subset(rows: list[dict], start: date, end: date) -> tuple[list[dict], list[int]]:
    selected = [(r, i) for i, r in enumerate(rows) if start <= date.fromisoformat(r["date"]) < end]
    return [x[0] for x in selected], [x[1] for x in selected]


def _training(rows: list[dict], end: date, horizon: int) -> tuple[list[dict], list[int]]:
    x, y = [], []
    for i, row in enumerate(rows):
        if date.fromisoformat(row["date"]) >= end or i + horizon >= len(rows):
            continue
        if date.fromisoformat(rows[i + horizon]["date"]) >= end:
            continue
        x.append(row); y.append(local_minimum_label(rows, i, horizon))
    return x, y


def _utility(labels: list[int], predictions: list[int], fp_cost: float) -> float:
    fp = sum(p == 1 and y == 0 for y, p in zip(labels, predictions))
    fn = sum(p == 0 and y == 1 for y, p in zip(labels, predictions))
    tp = sum(p == 1 and y == 1 for y, p in zip(labels, predictions))
    return (tp - fp_cost * fp - fn) / max(len(labels), 1)


def _fit(kind: str, names: list[str], parameter: float | int, x: list[dict], y: list[int]):
    if kind == "logistic_regression":
        return LogisticModel(names, l2=float(parameter)).fit(x, y)
    return ShallowBoostingModel(names, rounds=int(parameter)).fit(x, y)


def _metrics(rows: list[dict], indexes: list[int], events: list[dict], horizons: list[int], threshold: float,
             closing_lookback: int, rebound_bps: float, fp_cost: float) -> list[dict]:
    by_index = {int(e["index"]): e for e in events if e.get("policy_eligible")}
    result = []
    for horizon in horizons:
        for scenario in ("favorable_now", "window_closing"):
            eligible_indexes = [i for i in indexes if i >= horizon and i + horizon < len(rows)]
            if scenario == "favorable_now":
                labels = {i: local_minimum_label(rows, i, horizon) for i in eligible_indexes}
            else:
                labels = {i: window_closing_label(rows, i, closing_lookback, rebound_bps) for i in eligible_indexes}
            selected = [i for i, e in by_index.items() if i in labels and e["scenario"] == scenario]
            hits = [labels[i] for i in selected]
            random_rate = mean(labels.values()) if labels else 0.0
            hit_rate = mean(hits) if hits else 0.0
            benefits = []
            fast_costs, slow_costs = [], []
            for i in selected:
                local = [float(r["cbr_rate"]) for r in rows[i - horizon:i + horizon + 1]]
                benefits.append((mean(local) / float(rows[i]["cbr_rate"]) - 1) * 10_000)
                if i + 1 < len(rows):
                    fast_costs.append((float(rows[i + 1]["cbr_rate"]) / float(rows[i]["cbr_rate"]) - 1) * 10_000)
                if i + 3 < len(rows):
                    slow_costs.append((float(rows[i + 3]["cbr_rate"]) / float(rows[i]["cbr_rate"]) - 1) * 10_000)
            probabilities = [float(by_index[i]["confidence"]) if i in by_index and by_index[i]["scenario"] == scenario else 0.0 for i in eligible_indexes]
            brier = mean((p - labels[i]) ** 2 for p, i in zip(probabilities, eligible_indexes)) if eligible_indexes else 0.0
            fp = sum(not labels[i] for i in selected)
            result.append({
                "horizon": horizon, "scenario": scenario, "signals": len(selected),
                "hit_rate": round(hit_rate, 4), "random_hit_rate": round(random_rate, 4),
                "lift": round(hit_rate / random_rate, 4) if random_rate else None,
                "mean_timing_bps": round(mean(benefits), 2) if benefits else None,
                "fast_wait_cost_bps": round(mean(fast_costs), 2) if fast_costs else None,
                "slow_wait_cost_bps": round(mean(slow_costs), 2) if slow_costs else None,
                "false_positive_cost": round(fp * fp_cost, 2), "brier_score": round(brier, 5),
                "threshold": threshold,
            })
    return result


def walk_forward(rows: list[dict], config: dict) -> dict:
    if not rows:
        raise ValueError("no feature rows")
    model_cfg, wf, policy = config["model"], config["walk_forward"], config["policy"]
    primary_horizon = int(model_cfg["primary_horizon"])
    names = feature_names(rows)
    start = date.fromisoformat(rows[0]["date"])
    last = date.fromisoformat(rows[-1]["date"])
    train_end = add_months(start, int(wf["initial_train_months"]))
    folds, all_events, all_test_indexes = [], [], []
    while add_months(train_end, int(wf["validation_months"]) + int(wf["test_months"])) <= last:
        val_end = add_months(train_end, int(wf["validation_months"]))
        test_end = add_months(val_end, int(wf["test_months"]))
        x_train, y_train = _training(rows, train_end, primary_horizon)
        _, val_indexes = _subset(rows, train_end, val_end)
        choices = []
        candidates = [("logistic_regression", p) for p in model_cfg["logit_l2"]] + [("shallow_gradient_boosting", p) for p in model_cfg["boost_rounds"]]
        for kind, parameter in candidates:
            try:
                model = _fit(kind, names, parameter, x_train, y_train)
            except ValueError:
                continue
            probs = [model.predict(rows[i]) for i in val_indexes]
            labels = [local_minimum_label(rows, i, primary_horizon) for i in val_indexes]
            for threshold in model_cfg["thresholds"]:
                predictions = [int(p >= float(threshold)) for p in probs]
                choices.append((_utility(labels, predictions, float(model_cfg["false_positive_cost"])), kind, parameter, float(threshold)))
        if not choices:
            train_end = add_months(train_end, int(wf["step_months"])); continue
        validation_utility, kind, parameter, threshold = max(choices, key=lambda x: x[0])
        selection_model = _fit(kind, names, parameter, x_train, y_train)
        selection_probs = [selection_model.predict(r) if date.fromisoformat(r["date"]) < val_end else 0.0 for r in rows]
        rebound_choices = []
        for rb in policy["rebound_threshold_bps"]:
            val_raw = [e for e in raw_candidates(rows, selection_probs, threshold, float(rb), int(policy["window_closing_lookback"])) if int(e["index"]) in val_indexes]
            val_events = apply_policy(val_raw, int(policy["cooldown_sessions"]), int(policy["max_candidates_7d"]))
            selected_closing = {int(e["index"]) for e in val_events if e.get("policy_eligible") and e["scenario"] == "window_closing"}
            closing_labels = [window_closing_label(rows, i, int(policy["window_closing_lookback"]), float(rb)) for i in val_indexes]
            closing_predictions = [int(i in selected_closing) for i in val_indexes]
            rebound_choices.append((_utility(closing_labels, closing_predictions, float(model_cfg["false_positive_cost"])), float(rb)))
        closing_utility, rebound = max(rebound_choices, key=lambda x: x[0])
        x_final, y_final = _training(rows, val_end, primary_horizon)
        model = _fit(kind, names, parameter, x_final, y_final)
        _, test_indexes = _subset(rows, val_end, test_end)
        probs_all = [model.predict(r) if date.fromisoformat(r["date"]) < test_end else 0.0 for r in rows]
        candidates_raw = [e for e in raw_candidates(rows, probs_all, threshold, rebound, int(policy["window_closing_lookback"])) if int(e["index"]) in test_indexes]
        scenario_utility = {"favorable_now": validation_utility, "window_closing": closing_utility}
        events = apply_policy(candidates_raw, int(policy["cooldown_sessions"]), int(policy["max_candidates_7d"]), scenario_utility)
        fold_metrics = _metrics(rows, test_indexes, events, list(model_cfg["horizons"]), threshold,
                                int(policy["window_closing_lookback"]), rebound, float(model_cfg["false_positive_cost"]))
        folds.append({"train_end": train_end.isoformat(), "validation_end": val_end.isoformat(), "test_end": test_end.isoformat(),
                      "selected_model": kind, "selected_parameter": parameter, "threshold": threshold,
                      "rebound_threshold_bps": rebound, "validation_utility": round(validation_utility, 6),
                      "scenario_utility": {k: round(v, 6) for k, v in scenario_utility.items()}, "metrics": fold_metrics,
                      "events": events, "model": model.metadata()})
        all_events.extend(events); all_test_indexes.extend(test_indexes)
        train_end = add_months(train_end, int(wf["step_months"]))
    if not folds:
        raise ValueError("history is too short for 36m train + 6m validation + 3m test")
    aggregate = []
    for metric_horizon in model_cfg["horizons"]:
        for scenario in ("favorable_now", "window_closing"):
            group = [m for f in folds for m in f["metrics"] if m["horizon"] == metric_horizon and m["scenario"] == scenario]
            total = sum(m["signals"] for m in group)
            weighted_hit = sum(m["hit_rate"] * m["signals"] for m in group) / total if total else 0.0
            random_rate = mean(m["random_hit_rate"] for m in group) if group else 0.0
            bps_values = [m["mean_timing_bps"] for m in group if m["mean_timing_bps"] is not None]
            fast_values = [m["fast_wait_cost_bps"] for m in group if m["fast_wait_cost_bps"] is not None]
            slow_values = [m["slow_wait_cost_bps"] for m in group if m["slow_wait_cost_bps"] is not None]
            aggregate.append({"horizon": metric_horizon, "scenario": scenario, "signals": total,
                              "hit_rate": round(weighted_hit, 4), "random_hit_rate": round(random_rate, 4),
                              "lift": round(weighted_hit / random_rate, 4) if random_rate else None,
                              "mean_timing_bps": round(mean(bps_values), 2) if bps_values else None,
                              "fast_wait_cost_bps": round(mean(fast_values), 2) if fast_values else None,
                              "slow_wait_cost_bps": round(mean(slow_values), 2) if slow_values else None,
                              "false_positive_cost": round(sum(m["false_positive_cost"] for m in group), 2),
                              "brier_score": round(mean(m["brier_score"] for m in group), 5) if group else None})
    dates = sorted(date.fromisoformat(e["date"]) for e in all_events if e.get("policy_eligible"))
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    primary = [m for m in aggregate if m["horizon"] == primary_horizon]
    primary_lifts = {m["scenario"]: (m["lift"] or 0.0) for m in primary}
    robustness = []
    for key, label in (("cbr_rate", "CBR"), ("nbk_rate", "NBK"), ("moex_close", "MOEX")):
        resolved, hits = 0, 0
        for event in all_events:
            if not event.get("policy_eligible") or event["scenario"] != "favorable_now":
                continue
            i = int(event["index"])
            if i < primary_horizon or i + primary_horizon >= len(rows):
                continue
            current = float(rows[i][key])
            window = [float(r[key]) for r in rows[i-primary_horizon:i+primary_horizon+1]]
            hits += int(current == min(window)); resolved += 1
        baseline_labels = [local_minimum_label(rows, i, primary_horizon, key) for i in sorted(set(all_test_indexes)) if i >= primary_horizon and i + primary_horizon < len(rows)]
        baseline = mean(baseline_labels) if baseline_labels else 0.0
        hit_rate = hits / resolved if resolved else 0.0
        robustness.append({"source": label, "signals": resolved, "hit_rate": round(hit_rate, 4),
                           "random_hit_rate": round(baseline, 4), "lift": round(hit_rate / baseline, 4) if baseline else None})
    confidence_stability = [{"test_end": f["test_end"], "mean_confidence": round(mean(float(e["confidence"]) for e in f["events"]), 4) if f["events"] else None,
                             "candidates": len(f["events"])} for f in folds]
    return {"method": "expanding walk-forward 36m train / 6m validation / 3m untouched test / 3m step",
            "feature_names": names, "folds": folds, "metrics": aggregate,
            "frequency": {"eligible_signals": len(dates), "median_gap_days": sorted(gaps)[len(gaps)//2] if gaps else None,
                          "gap_std_days": round(pstdev(gaps), 2) if len(gaps) > 1 else 0.0,
                          "cluster_share_le_3d": round(mean(g <= 3 for g in gaps), 4) if gaps else 0.0},
            "primary_lifts": primary_lifts, "source_robustness": robustness, "confidence_stability": confidence_stability, "events": all_events,
            "last_selection": {k: folds[-1][k] for k in ("selected_model", "selected_parameter", "threshold", "rebound_threshold_bps")}}


def fit_live(rows: list[dict], as_of: date, backtest: dict, config: dict):
    horizon = int(config["model"]["primary_horizon"])
    x, y = _training(rows, as_of, horizon)
    selection = backtest["last_selection"]
    model = _fit(selection["selected_model"], backtest["feature_names"], selection["selected_parameter"], x, y)
    return model, float(selection["threshold"])
