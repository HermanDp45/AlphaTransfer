"""Causal rare65 evaluation and frozen H5 history-span selection."""
from pathlib import Path
import json, os, sys, warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from final_solution.training import core_experiment as core
from research_v4.robust_selection.evaluate import metric
from research_v4.final_sprint.common import select
from research_v4.continuation.oxr.assess import paired

HERE = Path(__file__).resolve().parent
TARGET_FREQUENCY = 0.65
RANK_LOOKBACK = 63
COOLDOWN = 2
POLICY = "rare65"
YEARS = (2024, 2025, 2026)


def save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def causal_rank(frame):
    """Rank each score against at most 63 strictly preceding scores."""
    q = frame.sort_values(["corridor", "date"]).reset_index(drop=True).copy()
    ranks = np.full(len(q), np.nan)
    for _, positions in q.groupby("corridor", sort=False).indices.items():
        positions = np.asarray(positions)
        scores = q.loc[positions, "probability"].to_numpy(float)
        for offset in range(len(scores)):
            past = scores[max(0, offset - RANK_LOOKBACK):offset]
            if len(past) >= 20 and np.isfinite(scores[offset]):
                ranks[positions[offset]] = (np.sum(past < scores[offset]) + 0.5 * np.sum(past == scores[offset])) / len(past)
    q["rank_score"] = ranks
    return q


def fit_rare65(validation):
    candidates = []
    for threshold in np.arange(0.0, 1.0001, 0.005):
        scored = validation.copy()
        scored["probability"] = scored.rank_score
        chosen, _ = select(scored, float(threshold), COOLDOWN)
        scored["candidate_signal"] = scored.index.isin(chosen)
        m = metric(scored, proper=False)
        candidates.append(dict(
            threshold=float(threshold), signals_per_corridor_week=m["signals_per_corridor_week"],
            absolute_frequency_error=abs(m["signals_per_corridor_week"] - TARGET_FREQUENCY),
            lift=m["lift"], forward_delta_bps=m["forward_delta_bps"], signals=m["signals"],
        ))
    # Frequency is the primary calibration target; labels only resolve equal-frequency choices.
    chosen = min(candidates, key=lambda row: (
        row["absolute_frequency_error"], -np.nan_to_num(row["lift"], nan=-1e9),
        -np.nan_to_num(row["forward_delta_bps"], nan=-1e9), -row["threshold"],
    ))
    return chosen, candidates


def prepare_fold(group, year):
    cutoff = pd.Timestamp(year, 1, 1)
    parts = {name: group[group.split.eq(name)].copy() for name in ("validation", "history", "test", "warmup")}
    assert len(parts["warmup"]) == RANK_LOOKBACK
    assert parts["warmup"].date.max() < parts["validation"].date.min()
    assert parts["validation"].label_available_date.lt(cutoff).all()
    calibrator = core.fit_platt_calibrator(parts["validation"].raw_probability.to_numpy(), parts["validation"].target)
    for frame in parts.values():
        frame["probability"] = core.apply_platt(calibrator, frame.raw_probability.to_numpy())

    calibration_sequence = causal_rank(pd.concat([
        parts["warmup"].assign(_role="warmup"), parts["validation"].assign(_role="validation")
    ], ignore_index=True))
    validation = calibration_sequence[calibration_sequence._role.eq("validation")].drop(columns="_role").reset_index(drop=True)
    assert validation.rank_score.notna().all()
    chosen, candidates = fit_rare65(validation)

    history_sequence = causal_rank(pd.concat([
        parts["warmup"].assign(_role="warmup"), parts["history"].assign(_role="history")
    ], ignore_index=True))
    history = history_sequence[history_sequence._role.eq("history")].drop(columns="_role").reset_index(drop=True)
    state_frame = history.copy()
    state_frame["probability"] = state_frame.rank_score
    _, initial_state = select(state_frame, chosen["threshold"], COOLDOWN)

    test_sequence = causal_rank(pd.concat([
        parts["warmup"].assign(_role="warmup"), parts["history"].assign(_role="history"),
        parts["test"].assign(_role="test")
    ], ignore_index=True))
    test = test_sequence[test_sequence._role.eq("test")].drop(columns="_role").reset_index(drop=True)
    scored = test.copy()
    scored["probability"] = scored.rank_score
    selected, _ = select(scored, chosen["threshold"], COOLDOWN, initial_state)
    test["candidate_signal"] = test.index.isin(selected)
    test["policy"] = POLICY
    policy = dict(
        year=year, threshold=chosen["threshold"], target_signals_per_week=TARGET_FREQUENCY,
        calibration_signals_per_week=chosen["signals_per_corridor_week"], cooldown=COOLDOWN,
        max_contacts_per_week=2, rank_lookback=RANK_LOOKBACK,
        rank_contract="strictly preceding scores only; warmup precedes calibration and test ranks use history plus prior test scores",
        selected_calibration_candidate=chosen, calibration_candidates=candidates,
        calibrator=dict(method=calibrator.method, intercept=calibrator.intercept, slope=calibrator.slope),
        initial_state=initial_state,
    )
    return test, policy


def candidate_summary(config_id, history_mode, aggregate, yearly):
    annual_frequency_ok = all(0.45 <= row["signals_per_corridor_week"] <= 0.85 for row in yearly)
    annual_lift_ok = all(row["lift"] >= 1.30 for row in yearly)
    annual_utility_ok = all(row["forward_delta_bps"] > 0 for row in yearly)
    aggregate_frequency_ok = 0.60 <= aggregate["signals_per_corridor_week"] <= 0.70
    qualified = aggregate_frequency_ok and annual_frequency_ok and annual_lift_ok and annual_utility_ok
    violations = (
        int(not aggregate_frequency_ok)
        + sum(not (0.45 <= row["signals_per_corridor_week"] <= 0.85) for row in yearly)
        + sum(row["lift"] < 1.30 for row in yearly)
        + sum(row["forward_delta_bps"] <= 0 for row in yearly)
    )
    return dict(
        config_id=config_id, history_mode=history_mode, policy=POLICY, qualified=qualified,
        aggregate_frequency_ok=aggregate_frequency_ok, annual_frequency_ok=annual_frequency_ok,
        annual_lift_ok=annual_lift_ok, annual_utility_ok=annual_utility_ok, violated_gates=violations,
        min_annual_frequency=min(row["signals_per_corridor_week"] for row in yearly),
        max_annual_frequency=max(row["signals_per_corridor_week"] for row in yearly),
        min_annual_forward_delta_bps=min(row["forward_delta_bps"] for row in yearly),
        min_annual_lift=min(row["lift"] for row in yearly), aggregate_lift=aggregate["lift"],
        aggregate_forward_delta_bps=aggregate["forward_delta_bps"], aggregate_brier=aggregate["brier"],
    )


def ranking_key(candidate):
    # Qualified candidates precede fallback candidates. Fallback first minimizes gate violations.
    return (
        not candidate["qualified"], candidate["violated_gates"] if not candidate["qualified"] else 0,
        -candidate["min_annual_forward_delta_bps"], -candidate["min_annual_lift"],
        -candidate["aggregate_lift"], -candidate["aggregate_forward_delta_bps"],
        candidate["aggregate_brier"], candidate["history_mode"] != "120m",
    )


def main():
    raw = pd.concat([
        pd.read_csv(HERE / "raw_predictions.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip"),
        pd.read_csv(HERE / "warmup.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip"),
    ], ignore_index=True)
    outputs, policies = [], []
    for (config_id, history_mode, year), group in raw.groupby(["config_id", "history_mode", "fold_test_year"]):
        prediction, policy = prepare_fold(group, int(year))
        outputs.append(prediction)
        policies.append(dict(config_id=config_id, history_mode=history_mode, policy=POLICY, **policy))
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_csv(HERE / "predictions.csv.gz", index=False)
    save(HERE / "policies.json", policies)

    annual_rows, aggregate_rows, candidates = [], [], []
    for (config_id, history_mode), group in predictions.groupby(["config_id", "history_mode"]):
        yearly = []
        for year, part in group.groupby("fold_test_year"):
            row = dict(config_id=config_id, history_mode=history_mode, policy=POLICY, year=int(year),
                       train_horizon=5, evaluation_horizon=5, evaluation_scope="KZT", **metric(part))
            annual_rows.append(row)
            yearly.append(row)
        aggregate = dict(config_id=config_id, history_mode=history_mode, policy=POLICY,
                         period="2024-2026", train_horizon=5, evaluation_horizon=5,
                         evaluation_scope="KZT", **metric(group))
        aggregate_rows.append(aggregate)
        candidates.append(candidate_summary(config_id, history_mode, aggregate, yearly))
    pd.DataFrame(annual_rows).to_csv(HERE / "by_year.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(HERE / "summary.csv", index=False)
    ranking = sorted(candidates, key=ranking_key)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    pd.DataFrame(ranking).to_csv(HERE / "ranking.csv", index=False)
    winner = ranking[0]
    selected_annual = [row for row in annual_rows if row["config_id"] == winner["config_id"]]
    selected_aggregate = next(row for row in aggregate_rows if row["config_id"] == winner["config_id"])
    selection = dict(
        selected_recipe=winner["history_mode"], selected_config_id=winner["config_id"], policy=POLICY,
        horizon=5, selection_status="qualified" if winner["qualified"] else "fallback_minimum_gate_violations",
        gates=dict(aggregate_frequency=[0.60, 0.70], annual_frequency=[0.45, 0.85],
                   minimum_annual_lift=1.30, positive_annual_forward_delta=True),
        ranking_rule=["maximum minimum annual forward delta", "maximum minimum annual lift",
                      "maximum aggregate lift", "maximum aggregate forward delta", "minimum Brier", "120m tie break"],
        selected_metrics=dict(aggregate=selected_aggregate, by_year=selected_annual), candidates=ranking,
        inference_note="Post-selection retrospective comparison on 2024-2026; final refit is not a new OOT test.",
    )
    save(HERE / "selection.json", selection)

    control = predictions[predictions.history_mode.eq("120m")]
    full = predictions[predictions.history_mode.eq("full_history")]
    intervals = []
    for period in (*YEARS, "all"):
        a = control if period == "all" else control[control.fold_test_year.eq(period)]
        b = full if period == "all" else full[full.fold_test_year.eq(period)]
        intervals.append(dict(
            baseline="120m", candidate="full_history", policy=POLICY, period=period,
            inference="paired monthly bootstrap; post-selection descriptive interval", **paired(a, b)
        ))
    pd.DataFrame(intervals).to_csv(HERE / "paired_intervals.csv", index=False)
    save(HERE / "completion.json", dict(status="evaluation_complete", selected_recipe=winner["history_mode"],
         selected_config_id=winner["config_id"], selection_status=selection["selection_status"],
         predictions_rows=len(predictions)))
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
