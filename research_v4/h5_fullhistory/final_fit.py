"""Final selected expanding-history H5 fit and portable source bundle."""
from pathlib import Path
import argparse, json, os, shutil, sys, time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from research_v4.h5_fullhistory import experiment as e
from research_v4.h5_fullhistory import evaluate
from final_solution.training import core_experiment as core
from research_v4.final_sprint.common import select
from final_solution.tabm_h5.features import build_features, read_sources

HERE = Path(__file__).resolve().parent
DEFAULT_PANEL = ROOT / "research_v4/h3_finalization/audit/latest_panel.pkl"
DEFAULT_SOURCE_RECEIPT = ROOT / "research_v4/h3_finalization/audit/latest_panel_receipt.json"


def load_training_panel(panel_path, source_receipt_path, cutoff):
    """Load the audited research panel or rebuild it from the portable bundle CSVs."""
    if panel_path is not None and panel_path.is_file():
        source_receipt = json.loads(source_receipt_path.read_text())
        expected_sha = source_receipt["latest_panel_sha256"]
        assert e.sha(panel_path) == expected_sha
        return pd.read_pickle(panel_path), "audited_pickle", expected_sha

    bundle_path = ROOT / "final_solution/tabm_h5/bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(
            "Neither the audited panel nor the portable H5 bundle is available; "
            "cannot reconstruct training features"
        )
    config = json.loads(bundle_path.read_text())
    base = bundle_path.parent
    for name, relative in config["source_paths"].items():
        source = base / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing portable H5 source {name}: {source}")
        expected = config.get("source_sha256", {}).get(name)
        if expected and e.sha(source) != expected:
            raise RuntimeError(f"Portable H5 source checksum mismatch: {name}")
    panel = build_features(read_sources(config["source_paths"], base), cutoff)
    panel_fingerprint = e.fingerprint(panel[["date", "corridor", "session_ordinal", "rub_per_unit", *e.FEATURES]])
    return panel, "portable_bundle_csv", panel_fingerprint


def make_parts(panel, cutoff):
    targeted = e.baseline.targeted(panel, 5)
    targeted = targeted[targeted.corridor.eq("KZT")].copy()
    calibration_start = cutoff - pd.DateOffset(months=12)
    eligible = targeted[targeted.target.notna() & targeted.label_available_date.notna()]
    train = eligible[eligible.date.ge(pd.Timestamp("2010-01-01")) & eligible.date.lt(calibration_start) &
                     eligible.label_available_date.lt(calibration_start)].copy()
    validation = eligible[eligible.date.ge(calibration_start) & eligible.date.lt(cutoff) &
                          eligible.label_available_date.lt(cutoff)].copy()
    history = targeted[targeted.date.ge(calibration_start) & targeted.date.lt(cutoff)].copy()
    prior = targeted[targeted.date.lt(calibration_start)]
    warmup = prior[prior.date.isin(sorted(prior.date.unique())[-63:])].copy()
    tail = history[history.target.isna() | history.label_available_date.isna() |
                   history.label_available_date.ge(cutoff)].copy()
    assert train.date.min() == pd.Timestamp("2010-01-01")
    assert train.label_available_date.max() < calibration_start
    assert validation.label_available_date.max() < cutoff
    assert warmup.date.nunique() == 63 and warmup.date.max() < validation.date.min()
    return targeted, dict(train=train, validation=validation, history=history, warmup=warmup, tail=tail)


def output(frame, raw, cutoff, split):
    extra = [column for column in ("ret1", "rub_per_unit") if column not in e.KEEP]
    result = frame[e.KEEP + extra].copy()
    result["raw_probability"] = raw
    if split in ("warmup", "tail"):
        result[["target", "forward_bps", "symmetric_bps", "regret_bps"]] = np.nan
    if split == "history":
        immature = result.label_available_date.isna() | result.label_available_date.ge(cutoff)
        result.loc[immature, ["target", "forward_bps", "symmetric_bps", "regret_bps"]] = np.nan
    result["config_id"] = "tabm_kzt_h5_fullhistory"
    result["history_mode"] = "full_history"
    result["train_horizon"] = 5
    result["cutoff"] = str(cutoff.date())
    result["fold_test_year"] = cutoff.year
    result["split"] = split
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    parser.add_argument("--cutoff", default="2026-09-05")
    parser.add_argument("--output", type=Path, default=HERE / "final_fit")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selection = json.loads((HERE / "selection.json").read_text())
    assert selection["selected_recipe"] == "full_history" and selection["policy"] == "rare65"
    cutoff = pd.Timestamp(args.cutoff)
    panel_path = args.panel if args.panel is not None else (DEFAULT_PANEL if DEFAULT_PANEL.is_file() else None)
    panel, panel_source, expected_panel_sha = load_training_panel(panel_path, args.source_receipt, cutoff)
    out, destination = args.output, args.output / "model"
    out.mkdir(parents=True, exist_ok=True)
    targeted, parts = make_parts(panel, cutoff)
    model = e.baseline.n.Neural(e.FEATURES, e.SEED)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        if (destination / "model.json").exists() and not args.force:
            model.load(destination)
            metadata = json.loads((destination / "model.json").read_text())
        else:
            metadata = model.fit(parts["train"], destination)
        frames = [output(parts[name], model.predict(parts[name]), cutoff, name)
                  for name in ("validation", "history", "warmup", "tail")]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(
        out / "raw_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    tables = {name: raw[raw.split.eq(name)].sort_values("date").copy()
              for name in ("validation", "history", "warmup", "tail")}
    calibrator = core.fit_platt_calibrator(tables["validation"].raw_probability.to_numpy(), tables["validation"].target)
    for table in tables.values():
        table["probability"] = core.apply_platt(calibrator, table.raw_probability.to_numpy())
    calibration_sequence = evaluate.causal_rank(pd.concat([
        tables["warmup"].assign(_role="warmup"), tables["validation"].assign(_role="validation")
    ], ignore_index=True))
    calibration_ranked = calibration_sequence[calibration_sequence._role.eq("validation")].drop(columns="_role").reset_index(drop=True)
    chosen, candidates = evaluate.fit_rare65(calibration_ranked)
    history_sequence = evaluate.causal_rank(pd.concat([
        tables["warmup"].assign(_role="warmup"), tables["history"].assign(_role="history")
    ], ignore_index=True))
    history_ranked = history_sequence[history_sequence._role.eq("history")].drop(columns="_role").reset_index(drop=True)
    selector_input = history_ranked.copy()
    selector_input["probability"] = selector_input.rank_score
    chosen_ids, selector_state = select(selector_input, chosen["threshold"], 2)
    history_ranked["candidate_signal"] = history_ranked.index.isin(chosen_ids)
    history_ranked.to_csv(
        out / "policy_history.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    past_scores = [{"date": str(row.date.date()), "session_ordinal": int(row.session_ordinal),
                    "probability": float(row.probability)} for row in history_ranked.tail(63).itertuples()]
    state = dict(schema_version=1, last_processed_date=str(history_ranked.date.max().date()),
                 last_processed_session=int(history_ranked.session_ordinal.max()),
                 past_scores=past_scores, selector_state=selector_state)
    e.save(out / "initial_state.json", state)

    receipt = dict(
        status="fitted", selected_recipe="full_history", policy="rare65", train_horizon=5,
        cutoff=cutoff, train_start=parts["train"].date.min(), train_end=parts["train"].date.max(),
        train_rows=len(parts["train"]), train_latest_label=parts["train"].label_available_date.max(),
        calibration_start=cutoff - pd.DateOffset(months=12), validation_start=parts["validation"].date.min(),
        validation_end=parts["validation"].date.max(), validation_latest_label=parts["validation"].label_available_date.max(),
        validation_rows=len(parts["validation"]), history_rows=len(parts["history"]),
        history_last=parts["history"].date.max(), warmup_dates=parts["warmup"].date.nunique(),
        tail_rows=len(parts["tail"]), latest_panel_date=targeted.date.max(), selected_epochs=metadata["selected_epochs"],
        training_fingerprint=e.fingerprint(parts["train"][["date", "corridor", *e.FEATURES, "target", "label_available_date"]]),
        weights_sha256=e.sha(destination / "weights.pt"), preprocessor_sha256=e.sha(destination / "preprocess.joblib"),
        panel_sha256=expected_panel_sha, panel_source=panel_source,
        raw_predictions_sha256=e.sha(out / "raw_predictions.csv.gz"),
    )
    e.save(out / "receipt.json", receipt)
    policy_receipt = dict(name="rare65", kind="rank", threshold=chosen["threshold"], window=63,
                          min_history=20, cooldown_sessions=2, max_contacts_per_week=2,
                          calibration_target_signals_per_week=0.65,
                          calibration_realized_signals_per_week=chosen["signals_per_corridor_week"],
                          selected_candidate=chosen, candidates=candidates)
    e.save(out / "policy_calibration_receipt.json", policy_receipt)
    source_paths = json.loads((ROOT / "research_v4/h3_finalization/audit/source_paths.json").read_text())
    bundle = dict(
        schema_version=1, profile_id="tabm_kzt_h5_expanding2010_rare65",
        model_id="tabm_kzt_h5_expanding2010_rare65_20260905", model_cutoff=str(cutoff.date()),
        train_horizon=5, corridor="KZT", history_mode="full_history", features=e.FEATURES,
        seed=e.SEED, architecture=e.baseline.n.ARCH, numerical_embeddings=e.baseline.n.EMBED,
        model=dict(weights="model/weights.pt", preprocessor="model/preprocess.joblib",
                   weights_sha256=receipt["weights_sha256"], preprocessor_sha256=receipt["preprocessor_sha256"]),
        calibration=dict(method=calibrator.method, intercept=calibrator.intercept, slope=calibrator.slope),
        policy=policy_receipt, initial_state="initial_state.json", source_paths=source_paths,
        source_sha256={name: e.sha(ROOT / path) for name, path in source_paths.items()},
        metadata=dict(training=receipt, selection=selection, historical_metrics_are_recipe_not_final_checkpoint=True,
                      final_refit_has_no_future_confirmatory_test=True, model_target="NOW_H5"),
    )
    e.save(out / "bundle.json", bundle)
    protocol = dict(created_unix=time.time(), selected_recipe="full_history", policy="rare65", cutoff=cutoff,
                    train_horizon=5, train_start="2010-01-01", calibration_months=12,
                    selection_sha256=e.sha(HERE / "selection.json"), panel_sha256=expected_panel_sha,
                    panel_source=panel_source,
                    source_receipt_sha256=e.sha(args.source_receipt) if args.source_receipt.is_file() else None,
                    code_sha256=e.sha(__file__))
    e.save(out / "protocol.json", protocol)
    reloaded = e.baseline.n.Neural(e.FEATURES, e.SEED)
    reloaded.load(destination)
    maximum_error = max(float(np.max(np.abs(reloaded.predict(parts[name]) - tables[name].raw_probability.to_numpy())))
                        if len(parts[name]) else 0.0 for name in tables)
    assert maximum_error == 0.0
    e.save(out / "verification.json", dict(status="PASS", selected_recipe="full_history", train_horizon=5,
           label_maturity=True, train_only_preprocessing=True, model_replay_exact=True,
           maximum_raw_replay_error=maximum_error, causal_policy=True, bundle_sha256=e.sha(out / "bundle.json")))
    print("FINAL H5 FIT PASS", json.dumps(receipt, default=str), flush=True)


if __name__ == "__main__":
    main()
