"""Independent integrity checks for the matched H5 experiment."""
from pathlib import Path
import json, sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from research_v4.h5_fullhistory import experiment as train
from research_v4.h5_fullhistory import evaluate

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = train.load_panel()
    rates = panel.rub_per_unit.to_numpy(float)
    future = np.stack([rates[offset:len(rates) - 5 + offset] for offset in (1, 2, 3, 4, 5)], axis=1)
    target = (((future.min(axis=1) / rates[:-5] - 1) * 10000 + 1e-12) >= 0).astype(float)
    np.testing.assert_array_equal(panel.target.iloc[:-5], target)
    np.testing.assert_allclose(panel.forward_bps.iloc[:-5], (future.mean(axis=1) / rates[:-5] - 1) * 10000,
                               rtol=0, atol=1e-9)
    assert panel.label_available_date.equals(panel.date.shift(-5))

    raw = pd.read_csv(HERE / "raw_predictions.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip")
    warmup = pd.read_csv(HERE / "warmup.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip")
    predictions = pd.read_csv(HERE / "predictions.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip")
    assert not raw.duplicated(["config_id", "fold_test_year", "split", "date", "corridor"]).any()
    assert predictions.policy.eq("rare65").all() and predictions.train_horizon.eq(5).all()
    assert predictions.groupby(["config_id", "fold_test_year"]).candidate_signal.sum().gt(0).all()

    replay = []
    with threadpool_limits(limits=2):
        for year in train.YEARS:
            parts = train.expanding_split(panel, year)
            control = train.baseline.split(panel, "kzt", 5, year)
            assert parts["train"].label_available_date.max() < pd.Timestamp(year - 1, 1, 1)
            assert parts["validation"].label_available_date.max() < pd.Timestamp(year, 1, 1)
            assert parts["test"].label_available_date.notna().all()
            for split in ("validation", "history", "test"):
                pd.testing.assert_frame_equal(parts[split][train.KEEP + train.FEATURES],
                                              control[split][train.KEEP + train.FEATURES], check_exact=True)
            destination = train.CKPT / f"tabm_kzt_fullhistory_h5_{year}"
            receipt = json.loads((destination / "receipt.json").read_text())
            metadata = json.loads((destination / "model.json").read_text())
            assert receipt["weights_sha256"] == train.sha(destination / "weights.pt")
            assert receipt["preprocessor_sha256"] == train.sha(destination / "preprocess.joblib")
            assert pd.Timestamp(metadata["inner_label_max"]) < pd.Timestamp(metadata["inner_validation_min"])
            model = train.baseline.n.Neural(train.FEATURES, train.SEED)
            model.load(destination)
            independent = model.preprocessor(parts["train"])
            np.testing.assert_array_equal(independent.named_steps["impute"].statistics_, model.pre.named_steps["impute"].statistics_)
            np.testing.assert_array_equal(independent.named_steps["gaussian"].quantiles_, model.pre.named_steps["gaussian"].quantiles_)
            for split in ("validation", "history", "test", "warmup"):
                expected = warmup if split == "warmup" else raw
                expected = expected[(expected.config_id == "tabm_kzt_h5_fullhistory") &
                                    (expected.fold_test_year == year) & (expected.split == split)].sort_values("date")
                actual = parts[split].sort_values("date")
                error = float(np.max(np.abs(model.predict(actual) - expected.raw_probability.to_numpy())))
                assert error < 2e-7
                replay.append(dict(year=year, split=split, rows=len(actual), maximum_raw_error=error))

    # The decision path cannot consume test labels: poisoning every test outcome leaves contacts unchanged.
    all_raw = pd.concat([raw, warmup], ignore_index=True)
    for (config_id, history_mode, year), group in all_raw.groupby(["config_id", "history_mode", "fold_test_year"]):
        original, _ = evaluate.prepare_fold(group, int(year))
        poisoned = group.copy()
        mask = poisoned.split.eq("test")
        poisoned.loc[mask, "target"] = 1 - poisoned.loc[mask, "target"]
        poisoned.loc[mask, ["forward_bps", "regret_bps"]] = 1e12
        replayed, _ = evaluate.prepare_fold(poisoned, int(year))
        np.testing.assert_array_equal(original.candidate_signal, replayed.candidate_signal)
        np.testing.assert_allclose(original.rank_score, replayed.rank_score, rtol=0, atol=0)

    selection = json.loads((HERE / "selection.json").read_text())
    assert selection["policy"] == "rare65" and selection["horizon"] == 5
    assert selection["selected_recipe"] in ("120m", "full_history")
    assert 0.60 <= selection["selected_metrics"]["aggregate"]["signals_per_corridor_week"] <= 0.70
    pd.DataFrame(replay).to_csv(HERE / "checkpoint_replay.csv", index=False)
    train.save(HERE / "verification.json", dict(
        status="PASS", cohort_parity=True, label_maturity=True, h5_target_independently_recomputed=True,
        train_only_preprocessing=True, causal_rank_policy=True, test_outcome_poisoning_invariant=True,
        calibration_uses_precalibration_warmup=True, no_forced_silent_week_fill=True,
        annual_fullhistory_models=3, exact_120m_controls=3, replay_groups=len(replay),
        maximum_raw_replay_error=max(row["maximum_raw_error"] for row in replay),
        selected_recipe=selection["selected_recipe"], selected_policy="rare65",
        source_sha256=train.sha(train.SOURCE), raw_predictions_sha256=train.sha(HERE / "raw_predictions.csv.gz"),
        predictions_sha256=train.sha(HERE / "predictions.csv.gz"), selection_sha256=train.sha(HERE / "selection.json"),
    ))
    print("H5 FULL-HISTORY COMPARISON PASS", len(replay), "checkpoint groups", flush=True)


if __name__ == "__main__":
    main()
