#!/usr/bin/env python3
"""Bounded TabM versus HGB benchmark using the incumbent temporal/policy protocol.

No web calls. All forecasts use frozen local feature snapshots. Outer 2026 is
diagnostic. Epoch selection uses only the last 63 unique dates of training;
the full two-year training set is then refit. Blending and calibration use
only the purged preceding year. No test label chooses a model or weight.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from tabm import TabM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from final_solution.training import core_experiment as core
from final_solution.training import train_and_evaluate as audit

H = 5
SEED = 20260904
YEARS = (2023, 2024, 2025, 2026)
CORRIDORS = {c: i for i, c in enumerate(("AMD", "KGS", "KZT", "TJS", "UZS"))}
MODEL_CONFIG = dict(n_blocks=2, d_block=128, dropout=0.1, k=16)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


class TabMClassifier:
    def __init__(self, features, date_lookup, output_dir, seed, max_epochs=200):
        self.features = features
        self.dates = date_lookup
        self.output_dir = output_dir
        self.seed = seed
        self.max_epochs = max_epochs
        self.fit_indexes = None
        self.model = None
        self.preprocess = None
        self.history = []
        self.metadata = {}

    def _preprocessor(self, frame):
        pre = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("gaussian", QuantileTransformer(n_quantiles=min(128, len(frame)),
                                             output_distribution="normal", random_state=self.seed)),
        ])
        pre.fit(frame[self.features])
        return pre

    def _encode(self, frame, pre):
        numeric = torch.as_tensor(pre.transform(frame[self.features]).astype(np.float32))
        cats = torch.as_tensor(frame.corridor.map(CORRIDORS).to_numpy(np.int64)[:, None])
        return numeric, cats

    def _new_model(self, seed):
        torch.manual_seed(seed)
        return TabM.make(n_num_features=len(self.features), cat_cardinalities=[5],
                         d_out=1, **MODEL_CONFIG)

    @staticmethod
    def _predict(model, encoded):
        model.eval()
        x, c = encoded
        result = []
        with torch.inference_mode():
            for start in range(0, len(x), 1024):
                logits = model(x[start:start + 1024], c[start:start + 1024]).squeeze(-1)
                result.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        return np.concatenate(result)

    def _train(self, frame, labels, pre, seed, epochs, validation=None):
        model = self._new_model(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
        x, c = self._encode(frame, pre)
        y = torch.as_tensor(np.asarray(labels, dtype=np.float32))
        generator = torch.Generator().manual_seed(seed)
        best = float("inf")
        best_epoch = 1
        for epoch in range(1, epochs + 1):
            model.train()
            permutation = torch.randperm(len(x), generator=generator)
            for inds in permutation.split(256):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x[inds], c[inds]).squeeze(-1)
                # The official TabM protocol trains each ensemble member's loss.
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y[inds, None].expand_as(logits))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            if validation is not None:
                valx, valy = validation
                valp = self._predict(model, valx)
                score = float(np.mean((valp - valy) ** 2))
                self.history.append({"epoch": epoch, "inner_validation_brier": score})
                if score < best - 1e-6:
                    best, best_epoch = score, epoch
                if epoch - best_epoch >= 20:
                    break
        return model, best_epoch, best

    def fit(self, frame, labels):
        indexes = tuple(frame.index)
        if self.fit_indexes is not None:
            if indexes != self.fit_indexes:
                raise ValueError("Attempted to reuse model on another train split")
            return self
        self.fit_indexes = indexes
        started = time.perf_counter()
        row_dates = self.dates.loc[frame.index]
        unique_dates = np.sort(row_dates.unique())
        cutoff = unique_dates[-63]
        last_allowed_inner = unique_dates[np.searchsorted(unique_dates, cutoff) - H - 1]
        inner_fit = row_dates <= last_allowed_inner
        inner_val = row_dates >= cutoff
        inner_pre = self._preprocessor(frame.loc[inner_fit])
        _, epochs, inner_brier = self._train(
            frame.loc[inner_fit], labels.loc[inner_fit], inner_pre, self.seed,
            self.max_epochs,
            (self._encode(frame.loc[inner_val], inner_pre), labels.loc[inner_val].to_numpy(float)),
        )
        self.preprocess = self._preprocessor(frame)
        self.model, _, _ = self._train(frame, labels, self.preprocess, self.seed + 1, epochs)
        self.metadata = {
            "seed": self.seed, "selected_epochs": epochs, "inner_validation_best_brier": inner_brier,
            "inner_train_end": pd.Timestamp(last_allowed_inner).date().isoformat(),
            "inner_validation_start": pd.Timestamp(cutoff).date().isoformat(),
            "inner_validation_dates": 63, "inner_purge_dates": H,
            "training_rows": len(frame), "training_unique_dates": len(unique_dates),
            "fit_seconds": time.perf_counter() - started,
            "parameter_count": sum(x.numel() for x in self.model.parameters()),
            "model_config": MODEL_CONFIG,
            "embedding": "No extra feature embeddings; basic official TabM architecture",
            "preprocessing": "train-only median impute and quantile-to-normal; native corridor category",
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "weights.pt")
        joblib.dump(self.preprocess, self.output_dir / "preprocess.joblib")
        write_json(self.output_dir / "model.json", {**self.metadata, "features": self.features})
        pd.DataFrame(self.history).to_csv(self.output_dir / "epoch_selection.csv", index=False)
        return self

    def predict_proba(self, frame):
        if self.model is None:
            raise ValueError("not fitted")
        p = self._predict(self.model, self._encode(frame, self.preprocess))
        return np.column_stack((1 - p, p))


class Prefit:
    def __init__(self, model, indexes):
        self.model, self.indexes = model, tuple(indexes)

    def fit(self, frame, labels):
        if tuple(frame.index) != self.indexes:
            raise ValueError("Wrong prefit train indexes")
        return self

    def predict_proba(self, frame):
        return self.model.predict_proba(frame)


class Blend:
    def __init__(self, tree, neural, neural_weight):
        self.tree, self.neural, self.weight = tree, neural, neural_weight

    def predict_proba(self, frame):
        return ((1 - self.weight) * self.tree.predict_proba(frame)
                + self.weight * self.neural.predict_proba(frame))


def delta_bootstrap(predictions, repetitions=10000):
    """Paired monthly moving units, retaining all corridor rows together."""
    rows = []
    dev = predictions[predictions.fold_test_year.isin([2023, 2024, 2025])].copy()
    for feature_set in ("base", "plus_cnyrub_basis"):
        base = dev[dev.config_id.eq("hgb_" + feature_set)][["date", "corridor", "target", "probability"]]
        for model in ("tabm_", "blend_"):
            cid = model + feature_set
            alt = dev[dev.config_id.eq(cid)][["date", "corridor", "probability"]]
            paired = base.merge(alt, on=["date", "corridor"], suffixes=("_base", "_alt"), validate="one_to_one")
            paired["delta"] = ((paired.probability_alt - paired.target)**2
                               - (paired.probability_base - paired.target)**2)
            dates = paired.groupby("date").delta.mean().reset_index()
            dates["year"] = dates.date.dt.year
            dates["month"] = dates.date.dt.to_period("M")
            rng = np.random.default_rng(SEED)
            sums, counts = np.zeros(repetitions), np.zeros(repetitions)
            for _, year in dates.groupby("year"):
                blocks = year.groupby("month").delta.agg(["sum", "count"])
                sampled = rng.integers(0, len(blocks), (repetitions, len(blocks)))
                sums += blocks["sum"].to_numpy()[sampled].sum(axis=1)
                counts += blocks["count"].to_numpy()[sampled].sum(axis=1)
            interval = np.quantile(sums/counts, [.025, .975])
            rows.append({"config_id": cid, "benchmark": "hgb_" + feature_set,
                         "delta_brier": dates.delta.mean(), "ci95_low": interval[0], "ci95_high": interval[1],
                         "unique_dates": len(dates), "bootstrap_repetitions": repetitions,
                         "status": "retrospective exploratory; year-stratified month-block interval"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    original_factory = core.make_model
    groups = audit.feature_groups(core)
    core.FEATURE_GROUPS.update(groups)
    started = time.perf_counter()
    panel = audit.build_feature_panel(ROOT, ROOT / "final_solution/data/normalized", core, audit.PROFILES["primary"])
    panel = core.add_target(panel, H)
    folds, predictions, fits = [], [], []
    write_json(output / "frozen_protocol.json", {
        "written_before_outer_evaluation": True, "status": "retrospective exploratory",
        "horizon": H, "years": args.years, "features": {k: groups[k] for k in ("base", "plus_cnyrub_basis")},
        "train_years": core.TRAIN_WINDOW_YEARS, "validation_years": 1,
        "test_never_used_for_epoch_calibration_blend_or_threshold_selection": True,
        "blend_weight_grid": [0, .25, .5, .75, 1], "model_config": MODEL_CONFIG,
        "max_epochs": args.max_epochs, "inner_early_stopping_patience": 20,
        "seed": SEED, "source_profile": "primary effective-date availability proxies",
        "dependencies": {p: importlib.metadata.version(p) for p in ["torch", "tabm", "numpy", "pandas", "scikit-learn", "scipy"]},
        "python": platform.python_version(), "code_sha256": digest(Path(__file__)),
        "cbr_sha256": digest(ROOT / "final_solution/data/cbr_daily.csv"),
        "cny_sha256": digest(ROOT / "final_solution/data/normalized/moex_cnyrub_tom.csv"),
        "fixing_sha256": digest(ROOT / "final_solution/data/normalized/moex_cny_fixing.csv"),
    })
    for feature_set in ("base", "plus_cnyrub_basis"):
        features = groups[feature_set]
        for year in args.years:
            train, validation, test = core.split_for_year(panel, H, year)
            columns = features + ["corridor"]
            tree = original_factory("hist_gradient_boosting", features)
            tree.fit(train[columns], train.target.astype(int))
            neural = TabMClassifier(features, panel.date, output / "artifacts" / f"{feature_set}_{year}",
                                    SEED + year, args.max_epochs)
            neural.fit(train[columns], train.target.astype(int))
            tv = tree.predict_proba(validation[columns])[:, 1]
            nv = neural.predict_proba(validation[columns])[:, 1]
            blend_grid = [(w, float(np.mean(((1-w)*tv + w*nv - validation.target.to_numpy())**2)))
                          for w in [0, .25, .5, .75, 1]]
            # Deterministic tie-breaking favours incumbent tree at equal score.
            weight, blend_brier = min(blend_grid, key=lambda pair: (pair[1], pair[0]))
            models = {"hgb": tree, "tabm": neural, "blend": Blend(tree, neural, weight)}
            for label, model in models.items():
                wrapped = Prefit(model, train.index)
                core.make_model = lambda kind, used_features, m=wrapped: m
                cid = label + "_" + feature_set
                # Reuse the exact incumbent calibration/cooldown implementation.
                experiment = core.Experiment(cid, "hist_gradient_boosting", feature_set)
                try:
                    metrics, oot, _ = core.run_fold(experiment, H, year, panel)
                finally:
                    core.make_model = original_factory
                for row in metrics:
                    row["model_kind"] = label
                oot["model_kind"] = label
                oot["feature_set"] = feature_set
                folds.extend(metrics)
                predictions.append(oot)
                print(json.dumps({"year": year, "model": cid,
                                  "brier": float(np.mean((oot.probability - oot.target)**2)),
                                  "candidate_count": int(oot.candidate_signal.sum()),
                                  "epoch": neural.metadata["selected_epochs"],
                                  "blend_neural_weight": weight}), flush=True)
            fits.append({"feature_set": feature_set, "fold_test_year": year, **neural.metadata,
                         "blend_neural_weight": weight, "blend_validation_raw_brier": blend_brier,
                         "blend_grid": blend_grid, "validation_rows": len(validation), "test_rows": len(test)})
            pd.DataFrame(folds).to_csv(output / "fold_corridor_metrics.csv", index=False)
            pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
            write_json(output / "training_details.json", fits)
    all_predictions = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.DataFrame(folds)
    summary_parts = []
    for period, years in [("development", [2023, 2024, 2025]), ("diagnostic_2026", [2026])]:
        part = fold_frame[fold_frame.fold_test_year.isin(years)]
        if part.empty:
            continue
        summary = core.aggregate_metrics(part)
        summary["period"] = period
        summary_parts.append(summary)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(output / "aggregate_metrics.csv", index=False)
    if set([2023, 2024, 2025]).issubset(args.years):
        delta_bootstrap(all_predictions).to_csv(output / "paired_brier_intervals.csv", index=False)
    selected_columns = ["period", "config_id", "model_brier", "candidate_signal_count",
                        "candidate_cell_standardized_lift", "candidate_cell_standardized_forward_bps_delta",
                        "candidate_cell_standardized_symmetric_bps_delta"]
    summary[selected_columns].to_csv(output / "scorecard.csv", index=False)
    write_json(output / "_SUCCESS.json", {"status": "complete", "seconds": time.perf_counter()-started,
                                           "output_hashes": {p.name: digest(p) for p in output.glob("*.csv")}})
    print(summary[selected_columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
