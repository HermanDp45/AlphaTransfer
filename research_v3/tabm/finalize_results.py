#!/usr/bin/env python3
"""Refresh 10k paired intervals and stability summaries without model refit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_tabm_benchmark import ROOT, delta_bootstrap, digest, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
pred = pd.read_csv(OUT / "predictions.csv", parse_dates=["date"])
delta_bootstrap(pred, repetitions=10000).to_csv(OUT / "paired_brier_intervals.csv", index=False)
cell_rows, year_rows = [], []
for feature in ("base", "plus_cnyrub_basis"):
    tree = pred[pred.config_id.eq("hgb_" + feature)]
    for label in ("tabm", "blend"):
        cid = label + "_" + feature
        neural = pred[pred.config_id.eq(cid)]
        matched = tree.merge(neural, on=["date", "corridor", "fold_test_year"],
                             suffixes=("_tree", "_neural"), validate="one_to_one")
        assert np.array_equal(matched.target_tree, matched.target_neural)
        matched["brier_tree"] = (matched.probability_tree - matched.target_tree)**2
        matched["brier_neural"] = (matched.probability_neural - matched.target_neural)**2
        matched["delta"] = matched.brier_neural - matched.brier_tree
        for (year, corridor), g in matched.groupby(["fold_test_year", "corridor"]):
            cell_rows.append(dict(config_id=cid, benchmark="hgb_"+feature, fold_test_year=int(year),
                                  corridor=corridor, dates=len(g), baseline_brier=g.brier_tree.mean(),
                                  model_brier=g.brier_neural.mean(), delta_brier=g.delta.mean()))
        for year, g in matched.groupby("fold_test_year"):
            year_rows.append(dict(config_id=cid, benchmark="hgb_"+feature, fold_test_year=int(year),
                                  first_date=g.date.min().date().isoformat(), last_date=g.date.max().date().isoformat(),
                                  unique_dates=g.date.nunique(), baseline_brier=g.brier_tree.mean(),
                                  model_brier=g.brier_neural.mean(), delta_brier=g.delta.mean(),
                                  relative_brier_improvement=-g.delta.mean()/g.brier_tree.mean(),
                                  improved_cells=int((g.groupby("corridor").delta.mean()<0).sum())))
pd.DataFrame(cell_rows).to_csv(OUT / "paired_cell_stability.csv", index=False)
pd.DataFrame(year_rows).to_csv(OUT / "paired_year_stability.csv", index=False)
incumbent = pd.concat([
    pd.read_csv(ROOT / "final_solution/model_bundle/development_h5_predictions.csv", parse_dates=["date"]),
    pd.read_csv(ROOT / "final_solution/model_bundle/diagnostic_2026_predictions.csv", parse_dates=["date"]),
], ignore_index=True)
parity = []
for cid in ("hgb_base", "hgb_plus_cnyrub_basis"):
    one = pred[pred.config_id.eq(cid)]
    two = incumbent[incumbent.config_id.eq(cid) & incumbent.horizon_cbr_rows_pub_proxy.eq(5)]
    joined = one.merge(two, on=["date", "corridor", "fold_test_year"], suffixes=("_new", "_old"), validate="one_to_one")
    max_error = float((joined.probability_new-joined.probability_old).abs().max())
    exact = bool(np.array_equal(joined.candidate_signal_new, joined.candidate_signal_old))
    assert len(joined)==len(one)==len(two)
    assert max_error<1e-10 and exact
    parity.append(dict(config_id=cid, rows=len(joined), max_probability_error=max_error,
                       candidate_masks_exact=exact, status="PASS"))
write_json(OUT / "incumbent_parity.json", parity)
receipt = json.loads((OUT / "_SUCCESS.json").read_text())
receipt.update(paired_brier_bootstrap_final_repetitions=10000, paired_brier_ci_refreshed_without_refit=True,
               readout_code_sha256=digest(Path(__file__)), current_runner_code_sha256=digest(HERE / "run_tabm_benchmark.py"),
               runner_change_since_training="Only default bootstrap repetitions changed from 2000 to 10000; no model refit")
receipt["output_hashes"] = {f.name:digest(f) for f in OUT.glob("*.csv")}
write_json(OUT / "_SUCCESS.json", receipt)
print(pd.read_csv(OUT / "paired_brier_intervals.csv").to_string(index=False))
print(pd.DataFrame(year_rows).to_string(index=False))
print(json.dumps(parity, indent=2))
