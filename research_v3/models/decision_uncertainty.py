#!/usr/bin/env python3
"""Exploratory policy intervals with jointly re-estimated random-day baselines.

All policies share block resamples. Cell (year x corridor) random-day means,
signal counts and signal outcomes are recomputed inside every bootstrap draw.
Thresholds are diagnostic selected frontiers, not independently validated rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import OUT, core

SEED = 20260905
CONFIGS = ("baseline_reproduction", "basis_train_120m", "annual_recent_calibration_3m")
METRICS = ("lift", "hit_rate", "forward_delta_bps", "symmetric_delta_bps", "regret_bps", "signals")


def safe_divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan, dtype=float),
                     where=denominator > 0)


def evaluate_sums(exposure, signal):
    """Inputs [..., cells, 5]: n, hit, forward, symmetric, regret sums."""
    count = signal[..., 0].sum(axis=-1)
    cell_mean = safe_divide(exposure[..., 1:], exposure[..., :1])
    expected_hit = (signal[..., 0] * cell_mean[..., 0]).sum(axis=-1)
    return dict(
        signals=count,
        hit_rate=safe_divide(signal[..., 1].sum(axis=-1), count),
        lift=safe_divide(signal[..., 1].sum(axis=-1), expected_hit),
        forward_delta_bps=safe_divide((signal[..., 2]-signal[..., 0]*cell_mean[..., 1]).sum(axis=-1), count),
        symmetric_delta_bps=safe_divide((signal[..., 3]-signal[..., 0]*cell_mean[..., 2]).sum(axis=-1), count),
        regret_bps=safe_divide(signal[..., 4].sum(axis=-1), count),
    )


def policies():
    frames = {}
    for cid in CONFIGS:
        path = OUT / f"{cid}_h5_predictions.csv.gz"
        frame = pd.read_csv(path, parse_dates=["date"]).sort_values(["date", "corridor"]).reset_index(drop=True)
        frames[cid] = frame
    base = frames[CONFIGS[0]].copy()
    masks = {}
    for cid, frame in frames.items():
        assert frame[["date", "corridor"]].equals(base[["date", "corridor"]])
        for field in ("target", "forward_bps", "symmetric_bps", "regret_bps"):
            assert np.allclose(frame[field], base[field], equal_nan=True), (cid, field)
        masks[f"{cid}::legacy"] = frame.candidate_signal.to_numpy(bool)
        for threshold in (.5, .65):
            # A single entrypoint at first available 2023 decision date. No test
            # label is used, but pre-2023 policy history is unavailable.
            selected = core.select_per_corridor_with_cooldown(frame, frame.probability.to_numpy(), threshold)
            masks[f"{cid}::threshold_{threshold:.2f}"] = frame.index.isin(selected)
    return base, masks


def aggregate_block_arrays(frame, masks):
    """Build common exposures and policy-specific sufficient statistics."""
    cells = list(frame.groupby(["fold_test_year", "corridor"]).groups)
    cell_index = {cell: i for i, cell in enumerate(cells)}
    months = sorted(frame.date.dt.to_period("M").unique())
    month_index = {month: i for i, month in enumerate(months)}
    ci = np.array([cell_index[(int(y), c)] for y, c in zip(frame.fold_test_year, frame.corridor)])
    bi = frame.date.dt.to_period("M").map(month_index).to_numpy(int)
    values = np.column_stack([np.ones(len(frame)), frame.target, frame.forward_bps,
                              frame.symmetric_bps, frame.regret_bps]).astype(float)
    assert np.isfinite(values).all()
    exposure = np.zeros((len(months), len(cells), 5))
    np.add.at(exposure, (bi, ci), values)
    signals = {}
    for pid, mask in masks.items():
        array = np.zeros_like(exposure)
        np.add.at(array, (bi, ci), values * np.asarray(mask)[:, None])
        signals[pid] = array
    return exposure, signals, months


def block_weights(months, repetitions):
    """A single weight matrix is shared by every policy and exposure baseline."""
    rng = np.random.default_rng(SEED)
    weights = np.zeros((repetitions, len(months)))
    years = np.array([month.year for month in months])
    for year in np.unique(years):
        indices = np.flatnonzero(years == year)
        # Multinomial block multiplicities are ordinary resampling with replacement.
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=repetitions)
    return weights


def cadence(frame, mask):
    selected = frame.index[np.asarray(mask)]
    coverage, weeks = [], 0.0
    for _, cell in frame.groupby(["fold_test_year", "corridor"]):
        weeks += ((cell.date.max()-cell.date.min()).days+1)/7
        coverage.append(core.cadence_diagnostics(cell, selected)["weeks_with_1_to_2_signals_share"])
    return dict(mean_signals_per_corridor_week=float(np.sum(mask))/weeks,
                min_cell_week_coverage=float(min(coverage)), mean_cell_week_coverage=float(np.mean(coverage)))


def run_readout(frame, masks, repetitions):
    exposure, signals, months = aggregate_block_arrays(frame, masks)
    weights = block_weights(months, repetitions)
    n_cells = exposure.shape[1]
    sampled_exposure = (weights @ exposure.reshape(len(months), -1)).reshape(repetitions, n_cells, 5)
    point_exposure = exposure.sum(axis=0)
    points, draws, rows = {}, {}, []
    supports = {}
    for pid, data in signals.items():
        sampled_signal = (weights @ data.reshape(len(months), -1)).reshape(repetitions, n_cells, 5)
        point = evaluate_sums(point_exposure, data.sum(axis=0))
        distribution = evaluate_sums(sampled_exposure, sampled_signal)
        points[pid], draws[pid] = point, distribution
        signal_frame = frame.loc[np.asarray(masks[pid])]
        support = dict(n_signal_dates=signal_frame.date.nunique(),
                       n_signal_month_blocks=signal_frame.date.dt.to_period("M").nunique())
        support["signal_support_status"] = ("sparse; no strong inference" if
            support["n_signal_dates"] < 20 or support["n_signal_month_blocks"] < 6
            else "descriptive retrospective interval")
        supports[pid] = support
        row = dict(policy_id=pid, n_unique_dates=frame.date.nunique(), n_month_blocks=len(months), **support,
                   bootstrap_repetitions=repetitions, **cadence(frame, masks[pid]))
        for metric in METRICS:
            values = distribution[metric]
            finite = np.isfinite(values)
            interval = np.quantile(values[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
            row.update({metric:float(point[metric]), metric+"_ci95_low":interval[0],
                        metric+"_ci95_high":interval[1], metric+"_valid_draws":int(finite.sum())})
        rows.append(row)
        row["zero_signal_draw_fraction"] = float(np.mean(distribution["signals"] == 0))
        row["interval_conditioning"] = ("conditional on nonempty signal draws; not nominal unconditional CI"
            if row["zero_signal_draw_fraction"] > 0 else "all draws nonempty")
    paired = []
    incumbent = "baseline_reproduction::legacy"
    pairs = [(pid, incumbent) for pid in signals if pid != incumbent]
    # Compare a threshold with its own model's inherited policy as well.
    pairs += [(pid, pid.split("::")[0]+"::legacy") for pid in signals
              if "threshold" in pid and not pid.startswith("baseline_reproduction::")]
    pairs += [(pid, "baseline_reproduction::"+pid.split("::")[1]) for pid in signals
              if "threshold" in pid and not pid.startswith("baseline_reproduction::")]
    for candidate, baseline in pairs:
        row = dict(candidate_policy=candidate, baseline_policy=baseline, bootstrap_repetitions=repetitions,
                   candidate_support_status=supports[candidate]["signal_support_status"],
                   baseline_support_status=supports[baseline]["signal_support_status"])
        for metric in METRICS:
            diff = draws[candidate][metric]-draws[baseline][metric]
            finite = np.isfinite(diff)
            interval = np.quantile(diff[finite], [.025, .975]) if finite.any() else [np.nan, np.nan]
            row.update({metric+"_delta":float(points[candidate][metric]-points[baseline][metric]),
                        metric+"_delta_ci95_low":interval[0], metric+"_delta_ci95_high":interval[1],
                        metric+"_paired_valid_draws":int(finite.sum())})
        paired.append(row)
    return rows, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    base, all_masks = policies()
    result_rows, paired_rows = [], []
    for track, years in (("development_2023_2025", [2023, 2024, 2025]), ("diagnostic_2026", [2026])):
        for trim in (0, 20):
            keep = base.fold_test_year.isin(years)
            if trim:
                date_sets = {}
                for year in years:
                    ds = sorted(base.loc[base.fold_test_year.eq(year), "date"].unique())
                    date_sets[year] = set(ds[trim:])
                keep &= pd.Series([d in date_sets.get(int(y), set()) for d,y in zip(base.date,base.fold_test_year)], index=base.index)
            frame = base.loc[keep].copy()
            masks = {pid:np.asarray(mask)[keep.to_numpy()] for pid,mask in all_masks.items()}
            rows, paired = run_readout(frame, masks, args.repetitions)
            context = dict(track=track, trim_first_dates_per_outer_fold=trim,
                           inferential_status="retrospective exploratory; thresholds post-selected; source availability proxies",
                           baseline_reestimated_each_draw=True)
            result_rows += [{**context, **row} for row in rows]
            paired_rows += [{**context, **row} for row in paired]
    result = pd.DataFrame(result_rows)
    result.to_csv(OUT / "decision_policy_intervals.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT / "decision_policy_paired_deltas.csv", index=False)
    manifest = dict(repetitions=args.repetitions, seed=SEED,
                    method="joint year-stratified calendar-month resampling; cell exposure baselines recomputed per draw",
                    pairing="identical block weights across policies, allowing different signal counts and dates",
                    fixed_threshold_initial_state="empty at first available 2023 decision; carried over observed prediction dates",
                    sensitivity="Drop first 20 matured test dates of each outer year without reselecting signals",
                    exposure="matured-label dates only; not full operational calendar",
                    limitations=["thresholds inspected after historical outcomes; no selection-corrected confirmation",
                                 "unlabelled year-end prediction dates and pre-2023 state unavailable",
                                 "random-day baseline and policy outcomes are official reference, not executable customer savings",
                                 "candidate-count change is part of policy tradeoff; not matched-count architecture comparison",
                                 "intervals use finite nonempty draws; sparse events cannot establish reliable precision",
                                 "20 signal dates / 6 signal months is a descriptive warning threshold, not a universal statistical theorem"],
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    output_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in [OUT/"decision_policy_intervals.csv",OUT/"decision_policy_paired_deltas.csv"]})
    (OUT / "decision_policy_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    columns = ["track","policy_id","signals","lift","lift_ci95_low","lift_ci95_high",
               "forward_delta_bps","forward_delta_bps_ci95_low","forward_delta_bps_ci95_high",
               "mean_signals_per_corridor_week","min_cell_week_coverage"]
    print(result[result.trim_first_dates_per_outer_fold.eq(0)][columns].to_string(index=False))


if __name__ == "__main__":
    main()
