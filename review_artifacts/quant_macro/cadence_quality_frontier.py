#!/usr/bin/env python3
"""Reproduce the post-hoc candidate cadence/quality frontier.

This diagnostic keeps only the corridor-policy layer.  For each development
fold it fits the frozen HGB+CNY-basis model, warms a past-score-only rolling
quantile controller on the prior year, and evaluates fixed quantiles on the
next OOT year.  The grid was inspected after the OOT results and is therefore
not a model-selection or confirmatory artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_ID = "hgb_plus_cnyrub_basis"
HORIZON = 5
DEVELOPMENT_YEARS = (2023, 2024, 2025)
QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
SCORE_WINDOW = 126
MINIMUM_HISTORY = 40
COOLDOWN_SESSIONS = 3


@dataclass(frozen=True)
class PreparedFold:
    year: int
    validation_history: pd.DataFrame
    validation_history_scores: np.ndarray
    test: pd.DataFrame
    test_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[2])
    parser.add_argument("--data-dir", type=Path, default=script.parent / "normalized")
    parser.add_argument("--output", type=Path, default=script.parent / "cadence_quality_frontier.csv")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rolling_select(
    frame: pd.DataFrame,
    scores: np.ndarray,
    quantile: float,
    history: list[float] | None = None,
    last_session: int = -10_000,
) -> tuple[np.ndarray, list[float], int]:
    ordered = frame.copy()
    ordered["_score"] = pd.Series(scores, index=frame.index)
    ordered = ordered.sort_values("date")
    score_history = list(history or [])
    selected: list[int] = []
    for index, score, session in ordered[["_score", "session_ordinal"]].itertuples(name=None):
        reference = score_history[-SCORE_WINDOW:]
        if len(reference) >= MINIMUM_HISTORY:
            threshold = float(np.quantile(reference, quantile, method="higher"))
            if score >= threshold and session - last_session > COOLDOWN_SESSIONS:
                selected.append(int(index))
                last_session = int(session)
        score_history.append(float(score))
    return np.asarray(selected, dtype=int), score_history[-SCORE_WINDOW:], last_session


def prepare_folds(repo_root: Path, data_dir: Path):
    core = load_module(
        "cadence_frontier_core",
        repo_root / "review_artifacts" / "experiments" / "clean_five_corridor_experiment.py",
    )
    quant = load_module(
        "cadence_frontier_quant",
        repo_root / "review_artifacts" / "quant_macro" / "quant_feature_ablation.py",
    )
    core.FEATURE_GROUPS.update(quant.feature_groups(core))
    experiment = next(item for item in quant.experiment_specs(core) if item.config_id == CONFIG_ID)
    panel = quant.build_feature_panel(repo_root, data_dir, core, quant.PROFILES["primary"])
    targeted = core.add_target(panel, HORIZON)
    features = core.FEATURE_GROUPS[experiment.feature_set]
    prepared: list[PreparedFold] = []
    for year in DEVELOPMENT_YEARS:
        train, validation, test = core.split_for_year(targeted, HORIZON, year)
        validation_history = targeted[
            (targeted["date"] >= pd.Timestamp(year - 1, 1, 1))
            & (targeted["date"] < pd.Timestamp(year, 1, 1))
        ].copy()
        model = core.make_model(experiment.model_kind, features)
        model.fit(train[features + ["corridor"]], train["target"].astype(int))
        validation_raw = model.predict_proba(validation[features + ["corridor"]])[:, 1]
        calibrator = core.fit_platt_calibrator(validation_raw, validation["target"])
        prepared.append(
            PreparedFold(
                year=year,
                validation_history=validation_history,
                validation_history_scores=core.apply_platt(
                    calibrator,
                    model.predict_proba(validation_history[features + ["corridor"]])[:, 1],
                ),
                test=test,
                test_scores=core.apply_platt(
                    calibrator,
                    model.predict_proba(test[features + ["corridor"]])[:, 1],
                ),
            )
        )
    return core, prepared


def evaluate_quantile(core, prepared: list[PreparedFold], quantile: float) -> dict[str, object]:
    oot_parts: list[pd.DataFrame] = []
    cadence_rows: list[dict[str, float]] = []
    for fold in prepared:
        selected: list[int] = []
        for corridor, warm_frame in fold.validation_history.groupby("corridor"):
            warm_positions = fold.validation_history.index.get_indexer(warm_frame.index)
            _, history, last_session = rolling_select(
                warm_frame,
                fold.validation_history_scores[warm_positions],
                quantile,
            )
            test_frame = fold.test[fold.test["corridor"].eq(corridor)]
            test_positions = fold.test.index.get_indexer(test_frame.index)
            corridor_selected, _, _ = rolling_select(
                test_frame,
                fold.test_scores[test_positions],
                quantile,
                history,
                last_session,
            )
            selected.extend(corridor_selected.tolist())
            diagnostics = core.cadence_diagnostics(test_frame, corridor_selected)
            span_days = max(1, (test_frame["date"].max() - test_frame["date"].min()).days + 1)
            cadence_rows.append(
                {
                    "signals_per_week": len(corridor_selected) / span_days * 7.0,
                    "weekly_fulfillment": diagnostics["weeks_with_1_to_2_signals_share"],
                    "silent_week_share": diagnostics["silent_week_share"],
                    "maximum_gap_days": diagnostics["maximum_calendar_gap_days_including_boundaries"],
                }
            )
        output = fold.test[
            ["date", "corridor", "target", "forward_bps", "symmetric_bps", "regret_bps"]
        ].copy()
        output["candidate_signal"] = output.index.isin(selected)
        output["fold_test_year"] = fold.year
        oot_parts.append(output)

    oot = pd.concat(oot_parts, ignore_index=True)
    cell_keys = ["fold_test_year", "corridor"]
    by_cell = oot.groupby(cell_keys)
    oot["cell_target_rate"] = by_cell["target"].transform("mean")
    for column in ("forward_bps", "symmetric_bps", "regret_bps"):
        oot[f"cell_{column}_mean"] = by_cell[column].transform("mean")
    signal = oot[oot["candidate_signal"]]
    expected_hits = float(signal["cell_target_rate"].sum())
    cadence = pd.DataFrame(cadence_rows)
    cell_lifts = []
    for _, cell in oot.groupby(cell_keys):
        cell_signal = cell[cell["candidate_signal"]]
        expected = float(cell_signal["cell_target_rate"].sum())
        cell_lifts.append(float(cell_signal["target"].sum() / expected) if expected else math.nan)
    return {
        "config_id": CONFIG_ID,
        "horizon_cbr_rows_pub_proxy": HORIZON,
        "development_years": ",".join(map(str, DEVELOPMENT_YEARS)),
        "policy_layer": "corridor_policy_candidate_only",
        "policy": "fixed_per_corridor_rolling_score_quantile",
        "score_window_sessions": SCORE_WINDOW,
        "minimum_history_sessions": MINIMUM_HISTORY,
        "cooldown_sessions": COOLDOWN_SESSIONS,
        "quantile": quantile,
        "fold_corridor_cells": len(cadence),
        "signal_count": len(signal),
        "cell_standardized_lift": float(signal["target"].sum() / expected_hits),
        "cell_standardized_hit_delta_pp": float(
            100.0 * (signal["target"] - signal["cell_target_rate"]).mean()
        ),
        "cell_standardized_symmetric_bps_delta": float(
            (signal["symmetric_bps"] - signal["cell_symmetric_bps_mean"]).mean()
        ),
        "cell_standardized_forward_bps_delta": float(
            (signal["forward_bps"] - signal["cell_forward_bps_mean"]).mean()
        ),
        "mean_signals_per_corridor_week": float(cadence["signals_per_week"].mean()),
        "minimum_fold_corridor_signals_per_week": float(cadence["signals_per_week"].min()),
        "maximum_fold_corridor_signals_per_week": float(cadence["signals_per_week"].max()),
        "mean_weekly_1_to_2_fulfillment": float(cadence["weekly_fulfillment"].mean()),
        "minimum_fold_corridor_weekly_1_to_2_fulfillment": float(
            cadence["weekly_fulfillment"].min()
        ),
        "mean_silent_week_share": float(cadence["silent_week_share"].mean()),
        "maximum_fold_corridor_silent_week_share": float(cadence["silent_week_share"].max()),
        "maximum_calendar_gap_days": float(cadence["maximum_gap_days"].max()),
        "minimum_fold_corridor_lift": float(np.nanmin(cell_lifts)),
        "passes_point_lift_1_3": bool(signal["target"].sum() / expected_hits >= 1.3),
        "passes_minimum_weekly_fulfillment_0_90": bool(cadence["weekly_fulfillment"].min() >= 0.90),
        "passes_both_point_screens": bool(
            signal["target"].sum() / expected_hits >= 1.3
            and cadence["weekly_fulfillment"].min() >= 0.90
        ),
        "partial_boundary_weeks_included": False,
        "selection_status": "posthoc_frontier_diagnostic_not_for_selection_or_go",
        "uncertainty_status": "point_estimates_only_use_block_bootstrap_before_claim",
        "warmup_status": "prior_year_backcast_with_frozen_fold_model_and_calibrator",
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    core, prepared = prepare_folds(repo_root, args.data_dir.resolve())
    result = pd.DataFrame(evaluate_quantile(core, prepared, quantile) for quantile in QUANTILES)
    result.to_csv(args.output.resolve(), index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
