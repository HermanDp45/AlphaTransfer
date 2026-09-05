"""Dependency-free historical decision API for V3 model/policy/behavior separation."""
from __future__ import annotations
import argparse
import csv
from datetime import date
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "final_solution"))
from alphatransfer_final.facts import historical_fact, factual_copy
from alphatransfer_final.behavior import build_behavior_preview
from alphatransfer_final.product import CURRENCY_COPY


def decision(as_of: date, model="baseline_reproduction", policy="selective", threshold=.50,
             corridors=None, context=None, horizon=5):
    """Only date, corridor, probability, stored policy and past session ids select.

    Future outcome columns are deliberately not parsed. Selective policy is an
    exploratory fixed-threshold alternative, not a promotion of a new forecast.
    """
    if Path(model).name != model or "/" in model or "\\" in model:
        raise ValueError("Model must be a registered local identifier")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability")
    if horizon not in (1,3,5,10,20):
        raise ValueError("Unsupported horizon")
    path = Path(__file__).parent / "models" / f"{model}_h{horizon}_predictions.csv.gz"
    if not path.is_file():
        raise ValueError(f"Model/horizon has no completed predictions: {model}, h={horizon}")
    receipt_path=path.with_name(f"{model}_h{horizon}_receipt.json")
    if not receipt_path.is_file():
        raise ValueError("Predictions lack a completed experiment receipt; rebuild this model first")
    receipt=json.loads(receipt_path.read_text())
    actual_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if receipt.get("status")!="complete" or receipt.get("predictions_sha256")!=actual_hash:
        raise ValueError("Prediction integrity check failed")
    relevant = set(CURRENCY_COPY if corridors is None else corridors)
    if not relevant or not relevant.issubset(CURRENCY_COPY):
        raise ValueError("Unknown or empty set of corridors")
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        for raw in csv.DictReader(source):
            if date.fromisoformat(raw["date"]) <= as_of and raw["corridor"] in relevant:
                rows.append({"date": raw["date"], "corridor": raw["corridor"], "probability": float(raw["probability"]), "session": int(raw["session_ordinal"]), "legacy_candidate": raw["candidate_signal"].lower() == "true"})
    rows.sort(key=lambda row: (row["date"], row["corridor"]))
    last = {}; current = []
    for row in rows:
        if policy == "legacy":
            selected = row["legacy_candidate"]
        elif policy == "selective":
            selected = row["probability"] >= threshold and row["session"] - last.get(row["corridor"], -10000) > 3
        else:
            raise ValueError("policy must be legacy or selective")
        if selected:
            last[row["corridor"]] = row["session"]
        if row["date"] == as_of.isoformat():
            fact = historical_fact(ROOT / "final_solution/data/cbr_daily.csv", row["corridor"], as_of)
            currency, country = CURRENCY_COPY[row["corridor"]]
            current.append({**row, "candidate": selected, "factual_evidence": fact, "copy": factual_copy(fact,currency,country)})
    if not current:
        raise ValueError("No evaluated decision date: choose a saved OOT date; no live forecasts are fabricated")
    selected = max((r for r in current if r["candidate"]), key=lambda r:r["probability"], default=None)
    behavior = build_behavior_preview(context, as_of)
    suppress = ["historical_research_preview", "no_executable_alpha_quote", "prospective_validation_pending", "source_rights_and_publication_clocks_require_verification"]
    if not selected:
        suppress.append("no_market_candidate")
    if behavior.get("preview_ready") is False or behavior["status"] in {"rejected", "unavailable"}:
        suppress.extend(f"behavior:{r}" for r in behavior["suppression_reasons"])
    if behavior.get("simulation"):
        suppress.append("behavior:synthetic_context_is_research_only")
    return {"schema_version":3, "status":"RESEARCH_PREVIEW_ONLY", "as_of":as_of.isoformat(), "model":model, "horizon_effective_cbr_rows":horizon,
            "policy":{"name":policy,"threshold":threshold if policy=="selective" else "prior-year validation", "cooldown_effective_rows":3,"frequency_is_a_cap_not_a_minimum":True,"selection_is_retrospective_exploratory":True},
            "source_predictions_sha256":actual_hash,
            "candidates":current,"selected":selected,"behavior":behavior,
            "eligible_to_send":False,"external_message_sent":False,"suppression_reasons":suppress,
            "metric_contract":{"forecast":"NOW hit probability, Brier/log-loss on paired dates", "policy":"hit lift, reference bps, risk and contact frequency jointly", "client":"simulation separately; causal value requires randomized bank data"}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of",type=date.fromisoformat,default=date(2025,12,16))
    p.add_argument("--model",default="baseline_reproduction")
    p.add_argument("--policy",choices=["legacy","selective"],default="selective")
    p.add_argument("--threshold",type=float,default=.5)
    p.add_argument("--horizon",type=int,choices=[1,3,5,10,20],default=5)
    p.add_argument("--corridors",default="AMD,KGS,KZT,TJS,UZS")
    p.add_argument("--client-context",type=Path)
    p.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"preview_output")
    a=p.parse_args(argv)
    if not 0<=a.threshold<=1:raise ValueError("Threshold must be a probability")
    context=json.loads(a.client_context.read_text()) if a.client_context else None
    result=decision(a.as_of,a.model,a.policy,a.threshold,a.corridors.split(","),context,a.horizon)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    out=a.output_dir/"decision.json";out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({"status":result["status"],"output":str(out.resolve()),"selected_corridor":result["selected"]["corridor"] if result["selected"] else None,"eligible_to_send":False},ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
