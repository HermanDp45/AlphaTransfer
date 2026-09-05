#!/usr/bin/env python3
"""Audit provider equivalence and write a compact numeric comparison table."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
keys=["date","corridor","fold_test_year"]

def compare(a,b,name):
    m=a.merge(b,on=keys,suffixes=("_a","_b"),validate="one_to_one")
    assert len(m)==len(a)==len(b),name
    diff=float(abs(m.probability_a-m.probability_b).max())
    assert diff < 1e-12,(name,diff)
    assert (m.candidate_signal_a==m.candidate_signal_b).all(),name
    return dict(check=name,rows=len(m),max_probability_difference=diff,identical_candidate_decisions=True)

def main():
    pred=pd.read_csv(HERE / "predictions.csv",parse_dates=["date"])
    stale=pd.read_csv(HERE / "sensitivity_predictions.csv",parse_dates=["date"])
    historical=pd.read_csv(REPO / "final_solution/model_bundle/development_h5_predictions.csv",parse_dates=["date"])
    checks=[]
    for cid in ("hgb_base","hgb_plus_cnyrub_basis"):
        a=pred[pred.config_id.eq(cid)&pred.fold_test_year.le(2025)]
        b=historical[historical.config_id.eq(cid)&historical.fold_test_year.le(2025)]
        checks.append(compare(a,b,"frozen_baseline_reproduced_"+cid))
    checks.append(compare(pred[pred.config_id.eq("hgb_plus_cnyrub_basis__fred")],pred[pred.config_id.eq("hgb_plus_cnyrub_basis__treasury")],"FRED_Treasury_primary_equivalence"))
    checks.append(compare(stale[stale.config_id.eq("hgb_plus_cnyrub_basis__fred_stale")],stale[stale.config_id.eq("hgb_plus_cnyrub_basis__treasury_stale")],"FRED_Treasury_stale_equivalence"))
    panel=pd.read_parquet(HERE / "feature_panel.parquet")
    violations={c:int((panel[c]>panel.date).sum()) for c in panel if c.startswith("v3_") and c.endswith("_available_date")}
    assert sum(violations.values())==0
    rows=[]
    for period in ("development","diagnostic_2026"):
        d=pd.read_csv(HERE/(period+"_metrics.csv"))
        d=pd.concat([d,pd.read_csv(HERE/(period+"_stale_metrics.csv"))])
        basis=d.set_index("config_id").loc["hgb_plus_cnyrub_basis"]
        for _,r in d.iterrows():
            rows.append(dict(period=period,config_id=r.config_id,brier=r.brier,brier_delta_vs_basis=r.brier-basis.brier,relative_brier_improvement_vs_basis=1-r.brier/basis.brier,candidate_hit_rate=r.candidate_hit_rate,candidate_lift=r.candidate_cell_standardized_lift,reference_gain_bps=r.candidate_cell_standardized_forward_bps_delta,gain_delta_bps_vs_basis=r.candidate_cell_standardized_forward_bps_delta-basis.candidate_cell_standardized_forward_bps_delta,weekly_1to2_coverage=r.mean_candidate_weeks_with_1_to_2_signals_share))
    pd.DataFrame(rows).to_csv(HERE / "RESULTS_COMPARISON.csv",index=False)
    verification=dict(status="passed",checks=checks,availability_violations=violations,limitations="Date lags are conservative proxies; this verifies join mechanics, not immutable source publication timestamps.")
    (HERE / "verification.json").write_text(json.dumps(verification,indent=2))
    paths=[p for p in HERE.rglob("*") if p.is_file() and p.name not in {"_SUCCESS.json"} and p.suffix!=".log" and "__pycache__" not in str(p)]
    success=dict(status="complete",research_status="retrospective_exploratory",unique_model_year_fits=58+(16 if (HERE/"long_combo/protocol.json").exists() else 0),production_promotions=0,outputs={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)})
    (HERE / "_SUCCESS.json").write_text(json.dumps(success,indent=2))
    print(json.dumps(verification,indent=2))

if __name__=="__main__":main()
