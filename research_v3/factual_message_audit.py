#!/usr/bin/env python3
"""Reproduce the historical-percentile evidence in METHODOLOGY_REVIEW.md."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from final_solution.training.core_experiment import build_panel

panel = build_panel(ROOT / "final_solution/data/cbr_daily.csv")
pred = pd.read_csv(ROOT / "final_solution/model_bundle/development_h5_predictions.csv", parse_dates=["date"])
pred = pred[pred.config_id.eq("hgb_plus_cnyrub_basis") & pred.candidate_signal]
merged = pred.merge(panel[["date", "corridor", "pr20", "pr60", "pr120"]],
                    on=["date", "corridor"], validate="one_to_one")
summary = {
    "config_id": "hgb_plus_cnyrub_basis",
    "period": "2023-2025 development OOT",
    "n_candidates": len(merged),
    "pr60_le_20pct": int((merged.pr60 <= .2).sum()),
    "pr60_gt_median": int((merged.pr60 > .5).sum()),
    "pr60_gt_80pct": int((merged.pr60 > .8).sum()),
    "interpretation": "NOW forecast confidence is not a historical-level predicate",
}
output = Path(__file__).parent / "factual_message_audit.json"
output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
merged[["date", "corridor", "probability", "pr20", "pr60", "pr120"]].to_csv(
    output.with_suffix(".csv"), index=False)
print(json.dumps(summary, ensure_ascii=False, indent=2))
