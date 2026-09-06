"""Matched H5 TabM comparison: 120 months versus expanding history from 2010."""
from pathlib import Path
import hashlib, json, os, pickle, sys, time, warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.tabm import experiment as baseline

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CKPT = HERE / "checkpoints"
SOURCE = baseline.SOURCE
FEATURES = baseline.FEATURES
KEEP = baseline.KEEP
SEED = baseline.SEED
HORIZON = 5
TRAIN_START = pd.Timestamp("2010-01-01")
YEARS = (2024, 2025, 2026)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def load_panel():
    views, _ = pickle.loads(SOURCE.read_bytes())
    panel = baseline.targeted(views["2010-01-01", 24, 1], HORIZON)
    return panel[panel.corridor.eq("KZT")].copy()


def expanding_split(panel, year):
    cutoff = pd.Timestamp(year, 1, 1)
    calibration_start = cutoff - pd.DateOffset(months=12)
    end = pd.Timestamp(year + 1, 1, 1)
    eligible = panel[panel.target.notna()]
    train = baseline.n.core.purge_tail(
        eligible[eligible.date.ge(TRAIN_START) & eligible.date.lt(calibration_start)], HORIZON
    )
    validation = baseline.n.core.purge_tail(
        eligible[eligible.date.ge(calibration_start) & eligible.date.lt(cutoff)], HORIZON
    )
    test = eligible[eligible.date.ge(cutoff) & eligible.date.lt(end)].copy()
    if end <= panel.date.max():
        test = baseline.n.core.purge_tail(test, HORIZON)
    history = panel[panel.date.ge(calibration_start) & panel.date.lt(cutoff)].copy()
    prior = panel[panel.date.lt(calibration_start)]
    warmup_dates = sorted(prior.date.unique())[-63:]
    warmup = prior[prior.date.isin(warmup_dates)].copy()
    control = baseline.split(panel, "kzt", HORIZON, year)
    assert train.date.min() == TRAIN_START
    assert train.label_available_date.max() < calibration_start
    assert validation.label_available_date.max() < cutoff
    assert test.label_available_date.notna().all()
    for name, part in (("validation", validation), ("history", history), ("test", test)):
        pd.testing.assert_frame_equal(part[KEEP + FEATURES], control[name][KEEP + FEATURES], check_exact=True)
    assert control["train"].index.isin(train.index).all()
    return dict(train=train, validation=validation, history=history, test=test, warmup=warmup)


def export(frame, scores, year, split, config_id):
    result = frame[KEEP].copy()
    result["raw_probability"] = scores
    if split == "warmup":
        result[["target", "forward_bps", "symmetric_bps", "regret_bps"]] = np.nan
    if split == "history":
        immature = result.label_available_date.ge(pd.Timestamp(year, 1, 1)) | result.label_available_date.isna()
        result.loc[immature, ["target", "forward_bps", "symmetric_bps", "regret_bps"]] = np.nan
    result["config_id"] = config_id
    result["history_mode"] = "full_history" if config_id.endswith("fullhistory") else "120m"
    result["train_horizon"] = HORIZON
    result["cutoff"] = f"{year}-01-01"
    result["fold_test_year"] = year
    result["split"] = split
    return result


def fit_fullhistory(panel, year):
    parts = expanding_split(panel, year)
    train = parts["train"]
    destination = CKPT / f"tabm_kzt_fullhistory_h5_{year}"
    model = baseline.n.Neural(FEATURES, SEED)
    contract = fingerprint(train[["date", "corridor", *FEATURES, "target", "label_available_date"]])
    if (destination / "receipt.json").exists():
        receipt = json.loads((destination / "receipt.json").read_text())
        assert receipt["training_fingerprint"] == contract
        assert receipt["source_sha256"] == sha(SOURCE)
        model.load(destination)
        metadata = json.loads((destination / "model.json").read_text())
    else:
        metadata = model.fit(train, destination)
    frames = []
    for split in ("validation", "history", "test", "warmup"):
        frames.append(export(parts[split], model.predict(parts[split]), year, split, "tabm_kzt_h5_fullhistory"))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / f"fullhistory_{year}_raw.csv.gz", index=False)
    control_rows = len(baseline.split(panel, "kzt", HORIZON, year)["train"])
    receipt = dict(
        config_id="tabm_kzt_h5_fullhistory", history_mode="full_history", year=year,
        train_horizon=HORIZON, train_start=train.date.min(), train_end=train.date.max(),
        train_rows=len(train), control_120m_rows=control_rows, extra_train_rows=len(train) - control_rows,
        train_latest_label=train.label_available_date.max(), calibration_start=pd.Timestamp(year - 1, 1, 1),
        validation_min=parts["validation"].date.min(), validation_max=parts["validation"].date.max(),
        validation_latest_label=parts["validation"].label_available_date.max(),
        test_min=parts["test"].date.min(), test_max=parts["test"].date.max(), test_rows=len(parts["test"]),
        warmup_rows=len(parts["warmup"]), warmup_last=parts["warmup"].date.max(),
        training_fingerprint=contract, selected_epochs=metadata["selected_epochs"],
        inner_latest_label=metadata["inner_label_max"], inner_validation_start=metadata["inner_validation_min"],
        weights_sha256=sha(destination / "weights.pt"), preprocessor_sha256=sha(destination / "preprocess.joblib"),
        source_sha256=sha(SOURCE), predictions_sha256=sha(OUT / f"fullhistory_{year}_raw.csv.gz"),
        feature_coverage={feature: float(train[feature].notna().mean()) for feature in FEATURES},
    )
    save(destination / "receipt.json", receipt)
    save(OUT / f"fullhistory_{year}_receipt.json", receipt)
    print(year, "full-history train", len(train), "extra", len(train) - control_rows,
          "epochs", metadata["selected_epochs"], flush=True)
    return result, receipt


def load_control(year):
    source = ROOT / "research_v4/robust_selection/tabm"
    raw = pd.read_csv(source / "raw_predictions.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip")
    warmup = pd.read_csv(source / "warmup.csv.gz", parse_dates=["date", "label_available_date"], float_precision="round_trip")
    mask = raw.config_id.eq("tabm_kzt") & raw.train_horizon.eq(HORIZON) & raw.fold_test_year.eq(year)
    wmask = warmup.config_id.eq("tabm_kzt") & warmup.train_horizon.eq(HORIZON) & warmup.fold_test_year.eq(year)
    result = pd.concat([raw[mask], warmup[wmask]], ignore_index=True)
    result["config_id"] = "tabm_kzt_h5_120m"
    result["history_mode"] = "120m"
    receipt = json.loads((source / "output" / f"tabm_kzt_h5_{year}_receipt.json").read_text())
    return result, receipt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = dict(
        experiment="matched_H5_history_span", years=YEARS, horizon=HORIZON, scope="KZT only",
        train_modes={"120m": "existing exact frozen control", "full_history": "expanding from 2010-01-01"},
        calibration_months=12, rank_warmup_sessions=63, seed=SEED, refit_seed=SEED + 1,
        features=FEATURES, architecture=baseline.n.ARCH, embeddings=baseline.n.EMBED, optimizer=baseline.n.OPT,
        leakage_contract="Actual fifth-next-session label must mature strictly before every boundary. Preprocessing is train-only. Validation/test cohorts are byte-identical across history modes.",
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in (SOURCE, baseline.FEATURE_FILE, Path(baseline.__file__), Path(baseline.n.__file__))},
    )
    save(HERE / "protocol.json", protocol)
    panel = load_panel()
    frames, receipts = [], []
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for year in YEARS:
            full, receipt = fit_fullhistory(panel, year)
            control, control_receipt = load_control(year)
            keys = ["fold_test_year", "split", "date", "corridor"]
            for split in ("validation", "history", "test", "warmup"):
                a = control[control.split.eq(split)].sort_values(keys).reset_index(drop=True)
                b = full[full.split.eq(split)].sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(a[keys], b[keys], check_exact=True, check_dtype=False)
                if split != "warmup":
                    pd.testing.assert_frame_equal(a[KEEP], b[KEEP], check_exact=True, check_dtype=False)
            frames.extend([control, full])
            receipts.append(dict(year=year, full_history=receipt, control_120m=control_receipt))
    combined = pd.concat(frames, ignore_index=True)
    combined[combined.split.ne("warmup")].to_csv(HERE / "raw_predictions.csv.gz", index=False)
    combined[combined.split.eq("warmup")].to_csv(HERE / "warmup.csv.gz", index=False)
    save(HERE / "receipts.json", receipts)
    save(HERE / "completion.json", dict(status="annual_fits_complete", full_history_neural_fits=3,
         control_models_reused=3, rows=len(combined), raw_sha256=sha(HERE / "raw_predictions.csv.gz"),
         warmup_sha256=sha(HERE / "warmup.csv.gz")))


if __name__ == "__main__":
    main()
