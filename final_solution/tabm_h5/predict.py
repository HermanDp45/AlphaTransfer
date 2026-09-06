"""Source-to-feature-to-NOW H5 production inference."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, read_sources
from .model import Predictor, load_config, sha
from .policy import binding, replay, validate_state


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(bundle_path, output_dir, as_of, *, mode="operational", state_in=None, state_out=None, source_config=None):
    config, base = load_config(bundle_path)
    if mode not in {"operational", "historical_smoke"}:
        raise ValueError("Invalid run mode")
    if isinstance(as_of, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of) is None:
        raise ValueError("run_as_of must be YYYY-MM-DD")
    now = pd.Timestamp(as_of)
    if now.tz is not None or now != now.normalize():
        raise ValueError("run_as_of must be a naive calendar date")
    cutoff = pd.Timestamp(config["model_cutoff"]).normalize()
    if now < cutoff:
        raise ValueError("Run as_of precedes the model availability cutoff")
    paths = config["source_paths"] if source_config is None else json.loads(Path(source_config).read_text(encoding="utf-8"))
    source_sha = {name: sha(base / Path(path)) for name, path in paths.items()}
    if source_config is None and config.get("source_sha256") and source_sha != config["source_sha256"]:
        raise ValueError("Frozen source checksum mismatch")
    features = build_features(read_sources(paths, base), now)
    initial = Path(state_in) if state_in else base / config["initial_state"]
    if not initial.is_file():
        raise ValueError("Explicit initial score/candidate state is required")
    state = json.loads(initial.read_text(encoding="utf-8"))
    validate_state(state, config)
    fresh = features if state["last_processed_session"] is None else features[features.session_ordinal.gt(state["last_processed_session"])]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if fresh.empty:
        decision = {"model": "h5", "status": "no_new_source_sessions", "candidate_signal": False, "rows": 0, "run_as_of": now.date().isoformat()}
        _json(output / "signal_decision.json", decision)
        return decision
    if mode == "operational" and fresh.date.lt(cutoff).any():
        raise ValueError("Pre-cutoff rows require historical_smoke mode")
    if mode == "operational" and features.iloc[-1].date < now:
        raise ValueError("No current-day feature observation; use historical_smoke for a dated snapshot")
    predictor = Predictor(config, base)
    raw, probability = predictor.predict(fresh)
    score_rows = [
        {"date": row.date.date().isoformat(), "corridor": "KZT", "session_ordinal": int(row.session_ordinal), "probability": float(score)}
        for row, score in zip(fresh.itertuples(), probability)
    ]
    decisions, next_state = replay(score_rows, config, state)
    for row in decisions:
        if row["reason_codes"] == ["NOW_H3_model_policy_pass"]:
            row["reason_codes"] = ["NOW_H5_model_policy_pass"]
    result = pd.DataFrame(decisions)
    result["raw_probability"] = raw
    result["train_horizon"] = 5
    result["model_id"] = config["model_id"]
    result["run_as_of"] = now.date().isoformat()
    result["chronology"] = np.where(fresh.date.to_numpy() < cutoff.to_datetime64(), "pre_model_cutoff_snapshot", "post_model_cutoff")
    result["rub_per_kzt_reference"] = fresh.rub_per_unit.to_numpy()
    result.to_csv(output / "predictions.csv", index=False)
    last = result.iloc[-1]
    snapshot = last.chronology == "pre_model_cutoff_snapshot"
    decision = {
        "model": "h5", "action": "infer", "status": "historical_snapshot_smoke" if mode == "historical_smoke" else "operational_model_candidate",
        "profile_id": config["profile_id"], "model_id": config["model_id"], "model_cutoff": config["model_cutoff"],
        "train_horizon": 5, "scenario": "NOW_H5", "corridor": "KZT", "run_as_of": now.date().isoformat(),
        "feature_date": str(last.date), "chronology": "pre_model_cutoff_snapshot" if snapshot else "post_model_cutoff",
        "candidate_signal": bool(last.candidate_signal), "probability": float(last.probability), "raw_probability": float(last.raw_probability),
        "rank_score": None if pd.isna(last.rank_score) else float(last.rank_score), "threshold": float(last.threshold),
        "reason_codes": list(last.reason_codes), "rows": int(len(result)), "NOW_contacts": int(result.candidate_signal.sum()),
        "authorized_contact": False, "external_messages_sent": 0, "is_oot_claim": False if mode == "historical_smoke" else None,
    }
    _json(output / "signal_decision.json", decision)
    _json(output / "next_state.json", next_state)
    if state_out:
        _json(Path(state_out), next_state)
    _json(output / "run_receipt.json", {
        "bundle_sha256": sha(bundle_path), "source_sha256": source_sha, "state_input_sha256": sha(initial),
        "predictions_sha256": sha(output / "predictions.csv"), "next_state_sha256": sha(output / "next_state.json"),
        "state_binding_sha256": binding(config), "rows": len(result), "run_as_of": now.date().isoformat(), "mode": mode,
        "research_imports_required": False,
    })
    return decision
