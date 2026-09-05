"""Reproducible retrospective learning-curve and model-capacity experiments.

Run from repository root: python research_v3/models/experiment.py --stage annual
V2 code and frozen inputs are imported, never rewritten. All selection remains
inside purged past folds. Annual test dates are identical to V2.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from final_solution.training import core_experiment as core
from final_solution.training import train_and_evaluate as ext

OUT = Path(__file__).resolve().parent
BASE_FEATURES = core.CORE_FEATURES + core.VOL_FEATURES
BASIS = "moex_cny_close_minus_fixing_same_session"
ORIGINAL_MAKE_MODEL = core.make_model
ORIGINAL_SPLIT = core.split_for_year


@dataclass(frozen=True)
class Spec:
    name: str
    months: int = 24  # zero means all available history
    features: str = "basis"
    model: str = "hgb2"
    extended: bool = False
    half_life_months: int = 0
    validation_months: int = 12
    historic_cny: bool = False


def specs():
    result = [Spec("baseline_reproduction")]
    for months in (3, 6, 12, 36, 60, 120, 0):
        result.append(Spec(f"basis_train_{months or 'all'}m", months=months, extended=months > 24 or months == 0))
    for months in (12, 24, 60, 120, 0):
        result.append(Spec(f"official_train_{months or 'all'}m", months=months, features="official", extended=months > 24 or months == 0))
    for model in ("hgb1", "hgb3", "hgb4", "hgb6", "logit", "extra_trees"):
        result.append(Spec(f"capacity_{model}", model=model))
    for months in (12, 24):
        result.append(Spec(f"decay_{months}m", months=0, extended=True, half_life_months=months))
    for fs in ("basis_normalized", "structural", "full"):
        result.append(Spec(f"features_{fs}", features=fs, model="hgb3"))
    for val in (1, 3, 6):
        result.append(Spec(f"annual_recent_calibration_{val}m", validation_months=val))
    for model in ("survival", "common_factor", "utility_mean", "utility_median", "utility_lower", "multiscale_ensemble"):
        result.append(Spec(f"objective_{model}", model=model))
    result.append(Spec("long_with_historical_cny", months=120, extended=True, historic_cny=True))
    result.append(Spec("long_short_ensemble", months=120, extended=True, model="long_short_ensemble"))
    return result


def panel(extended=False):
    cache = OUT / ("panel_extended.pkl" if extended else "panel_v2.pkl")
    receipt = cache.with_suffix(".receipt.json")
    fingerprint = input_fingerprint()
    if cache.exists() and receipt.exists() and json.loads(receipt.read_text()).get("fingerprint") == fingerprint:
        return pd.read_pickle(cache)
    old = ext.build_feature_panel(ROOT, ROOT / "final_solution/data/normalized", core, ext.PROFILES["primary"])
    if extended:
        p = core.build_panel(OUT / "data/cbr_extended.csv")
        external = [c for c in old if c not in p and c not in ("date", "corridor")]
        p = p.merge(old[["date", "corridor", *external]], on=["date", "corridor"], how="left", validate="one_to_one")
        # Preserve the V2 feature warm-up on test dates to make comparisons exact.
        assert np.allclose(p[p.date >= "2023-01-01"].ret1, old[old.date >= "2023-01-01"].ret1, equal_nan=True)
    else:
        p = old
    p = p.sort_values(["date", "corridor"]).reset_index(drop=True)
    p["basis_over_vol"] = p[BASIS] / p.vol20.clip(lower=0.001)
    p["basis_sign"] = np.sign(p[BASIS])
    p["basis_x_ret1"] = p[BASIS] * p.ret1
    for c, period in (("dom", 31), ("dow", 7), ("month", 12)):
        p[f"{c}_sin"] = np.sin(2 * np.pi * p[c] / period)
        p[f"{c}_cos"] = np.cos(2 * np.pi * p[c] / period)
    p["is_month_end"] = p.date.dt.is_month_end.astype(float)
    p["next_calendar_gap"] = p.date.dt.weekday.map({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 3, 6: 2})
    for lag in (1, 2, 3, 5):
        p[f"basis_lag{lag}"] = p.groupby("corridor")[BASIS].shift(lag)
    p.to_pickle(cache)
    receipt.write_text(json.dumps({"fingerprint":fingerprint,"rows":len(p)}))
    return p


def input_fingerprint():
    paths = [Path(__file__).resolve(), ROOT/"final_solution/training/core_experiment.py", ROOT/"final_solution/training/train_and_evaluate.py", ROOT/"final_solution/data/cbr_daily.csv", OUT/"data/cbr_extended.csv", *sorted((ROOT/"final_solution/data/normalized").glob("*.csv")), *sorted((ROOT/"research_v3/external_data/cny_history").glob("*.csv"))]
    digest=hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def feature_list(spec):
    if spec.features == "official":
        return BASE_FEATURES
    f = BASE_FEATURES + [BASIS]
    if spec.features in ("basis_normalized", "structural", "full"):
        f += ["basis_over_vol", "basis_sign", "basis_x_ret1", "basis_lag1", "basis_lag2", "basis_lag3", "basis_lag5"]
    if spec.features in ("structural", "full"):
        f += core.COMMON_FEATURES + ["moex_cnyrub_ret1", "moex_cnyrub_ret5", "moex_cnyrub_age_days", "moex_cny_fixing_age_days", "dow_sin", "dow_cos", "dom_sin", "dom_cos", "month_sin", "month_cos", "is_month_end", "next_calendar_gap"]
    if spec.features == "full":
        f += core.LEVEL_FEATURES + ["direct_corridor_market_minus_official_log_basis", "direct_corridor_market_ret1", "moex_rvi_ret5", "moex_imoex_ret5", "us_curve_2s10s", "eia_brent_ret5"]
    return list(dict.fromkeys(f))


def spec_panel(spec):
    p = panel(spec.extended)
    if spec.historic_cny:
        p = p.copy()
        frames = []
        for filename, prefix in (("moex_cnyrub_tom.csv", "moex_cnyrub"), ("moex_cny_fixing.csv", "moex_cny_fixing")):
            old = pd.read_csv(ROOT / "research_v3/external_data/cny_history" / filename)
            recent = pd.read_csv(ROOT / "final_solution/data/normalized" / filename)
            raw = pd.concat([old, recent], ignore_index=True).drop_duplicates("TRADEDATE", keep="last").sort_values("TRADEDATE").reset_index(drop=True)
            features = ext.add_availability(ext.market_features(raw, prefix), 1)
            joined = ext.asof_join(p[["date", "corridor"]], features, prefix, 7)
            frames.append(joined)
        first, fixing = frames
        p[BASIS] = (first.moex_cnyrub_log_level - fixing.moex_cny_fixing_log_level).where(first.moex_cnyrub_observation_date.eq(fixing.moex_cny_fixing_observation_date))
        # This variant uses only the original feature set plus newly available basis.
    return p


class WeightedModel:
    def __init__(self, model, frame, half_life_months):
        self.model, self.frame, self.half_life_months = model, frame, half_life_months

    def fit(self, x, y):
        dates = self.frame.loc[x.index, "date"]
        weights = np.power(0.5, (dates.max() - dates).dt.days / (30.4375 * self.half_life_months))
        weights /= weights.mean()
        self.model.fit(x, y, classifier__sample_weight=weights)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)


class StructuralModel:
    """Models task structure; future outcomes enter training labels only."""
    def __init__(self, spec, frame):
        self.spec, self.frame = spec, frame
        self.features = feature_list(spec)

    def base(self):
        m = ORIGINAL_MAKE_MODEL("hist_gradient_boosting", self.features)
        m.named_steps["classifier"].set_params(early_stopping=False)
        return m

    def fit(self, x, y):
        f = self.frame.loc[x.index]
        self.models = []
        if self.spec.model == "survival":
            # P(min_{1:h} r >= r0) = P(r1 >= r0) P(min_{1:h} r >= r0 | r1 >= r0).
            # This factorization does not expose realized first-day direction at inference.
            first_target = core.add_target(self.frame, 1).loc[x.index, "target"].astype(int)
            self.first = self.base().fit(x, first_target)
            self.conditional = self.base().fit(x[first_target.eq(1)], y[first_target.eq(1)])
        elif self.spec.model in ("multiscale_ensemble", "long_short_ensemble"):
            dates = f.date
            for months in ((24, 120) if self.spec.model == "long_short_ensemble" else (6, 12, 24)):
                take = dates >= dates.max() - pd.DateOffset(months=months)
                self.models.append(self.base().fit(x[take], y[take]))
        elif self.spec.model == "common_factor":
            # Five corridors share the RUB shock. Shrink their event targets towards
            # the mean event rate on that same training date, reducing idiosyncratic noise.
            target = y * 0.5 + pd.Series(y.to_numpy(), index=f.date).groupby(level=0).transform("mean").to_numpy() * 0.5
            self.regressor = self.base()
            self.regressor.steps[-1] = ("classifier", HistGradientBoostingRegressor(max_iter=120, max_depth=2, learning_rate=0.05, min_samples_leaf=40, l2_regularization=2, early_stopping=False, random_state=core.SEED))
            self.regressor.fit(x, target)
        else:
            self.regressor = self.base()
            quantile = {"utility_median": 0.5, "utility_lower": 0.2}.get(self.spec.model)
            params = dict(max_iter=120, max_depth=2, learning_rate=0.05, min_samples_leaf=60, l2_regularization=5, early_stopping=False, random_state=core.SEED)
            if quantile is not None:
                params.update(loss="quantile", quantile=quantile)
            self.regressor.steps[-1] = ("classifier", HistGradientBoostingRegressor(**params))
            # Winsorization bounds are fixed a priori, not chosen on test outcomes.
            self.regressor.fit(x, f.forward_bps.clip(-1000, 1000))
        return self

    def predict_proba(self, x):
        if self.spec.model == "survival":
            p = self.first.predict_proba(x)[:, 1] * self.conditional.predict_proba(x)[:, 1]
        elif self.spec.model in ("multiscale_ensemble", "long_short_ensemble"):
            p = np.mean([m.predict_proba(x)[:, 1] for m in self.models], axis=0)
        elif self.spec.model == "common_factor":
            p = self.regressor.predict(x)
        else:
            # Monotonic map only; a prior-year Platt layer then estimates NOW-hit probability.
            raw = self.regressor.predict(x)
            p = 1 / (1 + np.exp(-np.clip(raw / 100, -20, 20)))
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.column_stack([1-p, p])


def make_model(spec, frame):
    if spec.model in ("survival", "common_factor", "utility_mean", "utility_median", "utility_lower", "multiscale_ensemble", "long_short_ensemble"):
        return StructuralModel(spec, frame)
    features = feature_list(spec)
    model = ORIGINAL_MAKE_MODEL("hist_gradient_boosting", features)
    if spec.model.startswith("hgb"):
        depth = int(spec.model[3:])
        model.named_steps["classifier"].set_params(max_depth=depth, early_stopping=False)
    elif spec.model == "extra_trees":
        model.steps[-1] = ("classifier", ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=60, max_features=0.8, n_jobs=2, random_state=core.SEED))
    elif spec.model == "logit":
        model.steps[-1] = ("classifier", LogisticRegression(C=0.1, max_iter=2000))
    else:
        raise ValueError(spec.model)
    return WeightedModel(model, frame, spec.half_life_months) if spec.half_life_months else model


def temporal_split(frame, horizon, start, end, spec):
    val_start = start - pd.DateOffset(months=spec.validation_months)
    train_start = (val_start - pd.DateOffset(months=spec.months)) if spec.months else frame.date.min()
    eligible = frame[frame.target.notna()]
    train = core.purge_tail(eligible[(eligible.date >= train_start) & (eligible.date < val_start)], horizon)
    val = core.purge_tail(eligible[(eligible.date >= val_start) & (eligible.date < start)], horizon)
    test = eligible[(eligible.date >= start) & (eligible.date < end)].copy()
    if end <= frame.date.max():
        test = core.purge_tail(test, horizon)
    if len(train) < 150 or len(val) < 25:
        raise ValueError("Insufficient purged train/calibration history")
    assert train.date.max() < val.date.min() < test.date.min()
    return train, val, test


def run_annual(spec, horizon=5):
    p = core.add_target(spec_panel(spec), horizon)
    core.FEATURE_GROUPS[spec.name] = feature_list(spec)
    core.make_model = lambda kind, features: make_model(spec, p)
    core.split_for_year = lambda frame, h, year: temporal_split(frame, h, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1), spec)
    rows, pred = [], []
    try:
        for year in (2023, 2024, 2025, 2026):
            exp = core.Experiment(spec.name, "hist_gradient_boosting", spec.name)
            rr, pp, _ = core.run_fold(exp, horizon, year, p)
            pp["model_kind"] = spec.model
            pp["feature_set"] = spec.features
            pp = pp.merge(p[["date", "corridor", "rub_per_unit", "pr60", "ret1", "vol20", "session_ordinal", BASIS]], on=["date", "corridor"], validate="one_to_one")
            rows.extend(rr); pred.append(pp)
    finally:
        core.make_model, core.split_for_year = ORIGINAL_MAKE_MODEL, ORIGINAL_SPLIT
    pred = pd.concat(pred, ignore_index=True)
    pd.DataFrame(rows).to_csv(OUT / f"{spec.name}_h{horizon}_cells.csv", index=False)
    pred.to_csv(OUT / f"{spec.name}_h{horizon}_predictions.csv.gz", index=False)
    return pred


def summarize(pred):
    rows = []
    for track, group in (("development_2023_2025", pred[pred.fold_test_year <= 2025]), ("diagnostic_2026", pred[pred.fold_test_year == 2026])):
        for cid, g in group.groupby("config_id"):
            b = g.groupby(["fold_test_year", "corridor"]).agg(base_hit=("target", "mean"), base_forward=("forward_bps", "mean"), base_symmetric=("symmetric_bps", "mean"))
            s = g[g.candidate_signal].join(b, on=["fold_test_year", "corridor"])
            coverage = []
            observed_coverage = []
            for _, cell in g.groupby(["fold_test_year", "corridor"]):
                cadence = core.cadence_diagnostics(cell, cell.index[cell.candidate_signal])
                coverage.append(cadence["weeks_with_1_to_2_signals_share"])
                counts = cell.groupby(cell.date.dt.to_period("W-SUN")).candidate_signal.sum()
                counts = counts.reindex(pd.period_range(counts.index.min(),counts.index.max(),freq="W-SUN"),fill_value=0)
                observed_coverage.append(counts.between(1,2).mean())
            rows.append(dict(config_id=cid, track=track, rows=len(g), dates=g.date.nunique(), signals=len(s), brier=brier_score_loss(g.target, g.probability), log_loss=log_loss(g.target, g.probability, labels=[0,1]), auc=roc_auc_score(g.target, g.probability), hit_rate=s.target.mean(), lift=s.target.sum()/s.base_hit.sum() if len(s) else np.nan, forward_delta_bps=(s.forward_bps-s.base_forward).mean(), symmetric_bps=s.symmetric_bps.mean(), symmetric_delta_bps=(s.symmetric_bps-s.base_symmetric).mean(), regret_bps=s.regret_bps.mean(), min_cell_week_coverage=min(coverage), mean_cell_week_coverage=np.mean(coverage), min_observed_calendar_week_coverage=min(observed_coverage), mean_observed_calendar_week_coverage=np.mean(observed_coverage)))
    return pd.DataFrame(rows)


def run_dynamic(spec, horizon=5):
    """Monthly refits, with calibration separated in time and full annual score grid.

    No monthly test-tail purging: targets are used only after predictions are
    emitted, and later become train/validation labels only after own purge.
    The output keeps exactly the same annual eligible dates as the baseline.
    """
    p = core.add_target(spec_panel(spec), horizon)
    predictions, ledger = [], []
    last_state = {}
    for start in pd.date_range("2023-01-01", "2026-08-01", freq="MS"):
        end = start + pd.DateOffset(months=1)
        train, val, _ = temporal_split(p, horizon, start, end, spec)
        test = p[(p.date >= start) & (p.date < end) & p.target.notna()].copy()
        if test.empty:
            continue
        features = feature_list(spec)
        model = make_model(spec, p).fit(train[features + ["corridor"]], train.target.astype(int))
        vp = model.predict_proba(val[features + ["corridor"]])[:, 1]
        calibrator = core.fit_platt_calibrator(vp, val.target)
        scores = core.apply_platt(calibrator, model.predict_proba(test[features + ["corridor"]])[:, 1])
        thresholds, _, _ = core.choose_frequency_threshold(val, core.apply_platt(calibrator, vp))
        selected = core.select_per_corridor_with_cooldown(test, scores, thresholds, last_state)
        last_state.update(core.corridor_selection_state(test, selected))
        test["probability"] = scores
        test["candidate_signal"] = test.index.isin(selected)
        test["signal"] = False  # portfolio not a user; only corridor candidate policy evaluated
        test["fold_test_year"] = test.date.dt.year
        test["config_id"] = spec.name
        test["horizon_cbr_rows_pub_proxy"] = horizon
        predictions.append(test[["date", "corridor", "probability", "candidate_signal", "signal", "fold_test_year", "config_id", "horizon_cbr_rows_pub_proxy", "target", "forward_bps", "symmetric_bps", "regret_bps", "rub_per_unit", "pr60", "ret1", "vol20", "session_ordinal", BASIS]])
        ledger.append(dict(month=str(start.date()), train_start=str(train.date.min().date()), train_end=str(train.date.max().date()), train_dates=train.date.nunique(), validation_start=str(val.date.min().date()), validation_end=str(val.date.max().date()), validation_dates=val.date.nunique(), calibration_status=calibrator.status, thresholds=json.dumps(thresholds)))
    pred = pd.concat(predictions, ignore_index=True)
    baseline = pd.read_csv(OUT / f"baseline_reproduction_h{horizon}_predictions.csv.gz", usecols=["date", "corridor"], parse_dates=["date"])
    pred = pred.merge(baseline, on=["date", "corridor"], validate="one_to_one")
    pd.DataFrame(ledger).to_csv(OUT / f"{spec.name}_h{horizon}_fit_ledger.csv", index=False)
    pred.to_csv(OUT / f"{spec.name}_h{horizon}_predictions.csv.gz", index=False)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["annual", "dynamic", "summary"], default="annual")
    parser.add_argument("--only", default="")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    configurations = specs() if args.stage == "annual" else [
        Spec("monthly_train24_val3", validation_months=3),
        Spec("monthly_train24_val1", validation_months=1),
        Spec("monthly_train60_val3", months=60, extended=True, validation_months=3),
        Spec("monthly_structural_val3", features="structural", model="hgb3", validation_months=3),
    ]
    if args.only:
        configurations = [s for s in configurations if s.name in args.only.split(",")]
    if args.stage != "summary":
        (OUT / f"protocol_{args.stage}.json").write_text(json.dumps({"specs": [asdict(s) for s in configurations], "evaluation": "retrospective exploratory; 2023-25 development, inspected 2026 diagnostic", "seed": core.SEED, "python": platform.python_version(), "horizon": args.horizon, "source_data": "V2 snapshots plus isolated CBR 2010-2019 extension; market basis only available from 2020", "selection": "no test calibration; no random time-series cross-validation; no model promotion on diagnostic year"}, indent=2))
        for spec in configurations:
            target = OUT / f"{spec.name}_h{args.horizon}_predictions.csv.gz"
            receipt = OUT / f"{spec.name}_h{args.horizon}_receipt.json"
            expected = {"spec":asdict(spec),"input_code_fingerprint":input_fingerprint(),"stage":args.stage,"horizon":args.horizon}
            old_receipt=json.loads(receipt.read_text()) if receipt.exists() else {}
            if target.exists() and all(old_receipt.get(k)==v for k,v in expected.items()) and old_receipt.get("predictions_sha256")==hashlib.sha256(target.read_bytes()).hexdigest():
                print("CACHED", spec.name, flush=True); continue
            start = time.time()
            pred = (run_annual if args.stage == "annual" else run_dynamic)(spec, args.horizon)
            receipt.write_text(json.dumps({**expected,"predictions_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"rows":len(pred),"python":platform.python_version(),"status":"complete"},indent=2))
            print(spec.name, round(time.time()-start, 1), summarize(pred).to_dict("records"), flush=True)
    preds = [pd.read_csv(f, parse_dates=["date"]) for f in sorted(OUT.glob(f"*_h{args.horizon}_predictions.csv.gz"))]
    if preds:
        summary = summarize(pd.concat(preds, ignore_index=True))
        summary.to_csv(OUT / f"summary_h{args.horizon}.csv", index=False)
        print(summary.sort_values(["track", "brier"]).to_string(index=False))


if __name__ == "__main__":
    main()
