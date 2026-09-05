"""Paired date-block uncertainty and honest policy frontiers for all V3 trials."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import OUT, ROOT, core, summarize, specs

SEED = 20260904
REPS = 10000


def bootstrap_paired(joined, reps=REPS, block="month"):
    """Resample complete calendar blocks jointly across all five corridors."""
    j = joined.copy()
    j["delta"] = (j.probability - j.target) ** 2 - (j.base_probability - j.target) ** 2
    if block == "month":
        j["block"] = j.date.dt.to_period("M").astype(str)
    else:
        ordinal = {d: i for i, d in enumerate(sorted(j.date.unique()))}
        j["block"] = j.date.map(ordinal) // int(block)
    b = j.groupby(["fold_test_year", "block"]).agg(total=("delta", "sum"), n=("delta", "size"))
    rng = np.random.default_rng(SEED)
    total = np.zeros(reps); count = np.zeros(reps)
    for _, g in b.groupby(level=0):
        samples = rng.integers(0, len(g), (reps, len(g)))
        total += g.total.to_numpy()[samples].sum(axis=1)
        count += g.n.to_numpy()[samples].sum(axis=1)
    dist = total/count
    return dict(delta_brier=j.delta.mean(), ci95_low=np.quantile(dist, .025), ci95_high=np.quantile(dist, .975), improved_years=int((j.groupby("fold_test_year").delta.mean()<0).sum()), improved_cells=int((j.groupby(["fold_test_year", "corridor"]).delta.mean()<0).sum()), n_blocks=len(b))


def policy_stats(g, selected):
    s = g[selected].copy()
    b = g.groupby(["fold_test_year", "corridor"]).agg(base_hit=("target", "mean"), base_forward=("forward_bps", "mean"))
    s = s.join(b, on=["fold_test_year", "corridor"])
    calendar_weeks = sum((cell.date.max()-cell.date.min()).days/7+1/7 for _, cell in g.groupby(["fold_test_year", "corridor"]))
    coverage = [core.cadence_diagnostics(cell, s.index)["weeks_with_1_to_2_signals_share"] for _,cell in g.groupby(["fold_test_year", "corridor"])]
    return dict(signals=len(s), mean_per_corridor_week=len(s)/calendar_weeks, hit_rate=s.target.mean(), lift=s.target.sum()/s.base_hit.sum() if len(s) else np.nan, forward_delta_bps=(s.forward_bps-s.base_forward).mean(), symmetric_bps=s.symmetric_bps.mean(), regret_bps=s.regret_bps.mean(), min_cell_week_coverage=min(coverage), mean_cell_week_coverage=np.mean(coverage))


def frontier(preds):
    results=[]
    # These thresholds are a full diagnostic frontier, not independently validated policies.
    for cid in ["baseline_reproduction", "basis_train_120m", "annual_recent_calibration_3m"]:
        p = preds[cid]
        for track, g in [("development_2023_2025",p[p.fold_test_year<=2025]),("diagnostic_2026",p[p.fold_test_year==2026])]:
            for threshold in (0., .30, .35, .40, .45, .50, .55, .60, .65, .70):
                selected = core.select_per_corridor_with_cooldown(g, g.probability.to_numpy(), threshold)
                mask=g.index.isin(selected)
                results.append(dict(config_id=cid,track=track,policy=f"fixed_probability_{threshold:.2f}", threshold=threshold,**policy_stats(g,mask)))
            results.append(dict(config_id=cid,track=track,policy="legacy_validation_cadence", threshold=np.nan,**policy_stats(g,g.candidate_signal)))
            # Lower-quintile fact is a separate semantic scenario, not a prediction gain.
            results.append(dict(config_id=cid,track=track,policy="legacy_plus_historical_low_fact",threshold=np.nan,**policy_stats(g,g.candidate_signal & g.pr60.le(.2))))
    return pd.DataFrame(results)


def forward_policy_ci(g, reps=REPS):
    """Month-block uncertainty, conditional on frozen signals and candidate exposure."""
    g = g.copy()
    bases=g.groupby(["fold_test_year","corridor"]).agg(base_hit=("target","mean"),base_forward=("forward_bps","mean"))
    g=g.join(bases,on=["fold_test_year","corridor"])
    g["hits"]=g.candidate_signal*g.target
    g["expected"]=g.candidate_signal*g.base_hit
    g["advantage"]=g.candidate_signal*(g.forward_bps-g.base_forward)
    g["month"]=g.date.dt.to_period("M").astype(str)
    b=g.groupby(["fold_test_year","month"])[["hits","expected","advantage","candidate_signal"]].sum()
    total=np.zeros((reps,4));rng=np.random.default_rng(SEED)
    for _,cell in b.groupby(level=0):
        a=cell.to_numpy(float); total+=a[rng.integers(0,len(a),(reps,len(a)))].sum(axis=1)
    lift=total[:,0]/np.maximum(total[:,1],1e-9);bps=total[:,2]/np.maximum(total[:,3],1)
    return dict(lift_ci95_low=np.quantile(lift,.025),lift_ci95_high=np.quantile(lift,.975),forward_ci95_low=np.quantile(bps,.025),forward_ci95_high=np.quantile(bps,.975),note="Exploratory, selection not included; fold-corridor baselines held fixed")


def main():
    preds={f.name.split("_h5_predictions")[0]:pd.read_csv(f,parse_dates=["date"]) for f in OUT.glob("*_h5_predictions.csv.gz")}
    base=preds["baseline_reproduction"][["date","corridor","probability","target","candidate_signal"]].rename(columns={"probability":"base_probability","target":"base_target","candidate_signal":"base_candidate"})
    audits=[]
    for cid,p in preds.items():
        if cid=="baseline_reproduction":continue
        j=p.merge(base,on=["date","corridor"],validate="one_to_one")
        assert len(j)==len(base), (cid,len(j),len(base))
        assert np.array_equal(j.target,j.base_target)
        for track,g in [("development_2023_2025",j[j.fold_test_year<=2025]),("diagnostic_2026",j[j.fold_test_year==2026])]:
            for block in ("month",20,60):
                audits.append(dict(config_id=cid,track=track,block=str(block),**bootstrap_paired(g,block=block)))
    pd.DataFrame(audits).to_csv(OUT/"paired_uncertainty.csv",index=False)
    summary=summarize(pd.concat(preds.values(),ignore_index=True))
    summary.to_csv(OUT/"summary_h5.csv",index=False)
    frontier(preds).to_csv(OUT/"risk_coverage_frontier.csv",index=False)
    ci=[]
    for cid in ("baseline_reproduction","basis_train_120m","annual_recent_calibration_3m"):
        for track,g in [("development_2023_2025",preds[cid][preds[cid].fold_test_year<=2025]),("diagnostic_2026",preds[cid][preds[cid].fold_test_year==2026])]:
            ci.append(dict(config_id=cid,track=track,**forward_policy_ci(g)))
    pd.DataFrame(ci).to_csv(OUT/"policy_uncertainty.csv",index=False)
    cells=[]
    for cid,p in preds.items():
        for (year,corridor),g in p.groupby(["fold_test_year","corridor"]):
            cells.append(dict(config_id=cid,fold_test_year=year,corridor=corridor,brier=((g.probability-g.target)**2).mean(),**policy_stats(g,g.candidate_signal)))
    pd.DataFrame(cells).to_csv(OUT/"all_cells.csv",index=False)
    combo_path=ROOT/"research_v3/external_data/long_combo/predictions.csv.gz"
    if combo_path.exists():
        combo=pd.read_csv(combo_path,parse_dates=["date"])
        comparisons=[]
        for cid,g in combo.groupby("config_id"):
            g=g.merge(base,on=["date","corridor"],validate="one_to_one")
            for track,part in [("development_2023_2025",g[g.fold_test_year<=2025]),("diagnostic_2026",g[g.fold_test_year==2026])]:
                comparisons.append(dict(config_id=cid,track=track,baseline="baseline_reproduction",**bootstrap_paired(part)))
        pd.DataFrame(comparisons).to_csv(OUT/"external_combo_vs_incumbent_ci.csv",index=False)
    # Numerical reproduction is established by prediction-level parity, not rounded metrics.
    old=pd.concat([pd.read_csv(ROOT/"final_solution/model_bundle/development_h5_predictions.csv",usecols=["config_id","date","corridor","probability","candidate_signal"]),pd.read_csv(ROOT/"final_solution/model_bundle/diagnostic_2026_predictions.csv",usecols=["config_id","date","corridor","probability","candidate_signal"])])
    old=old[old.config_id.eq("hgb_plus_cnyrub_basis")];old["date"]=pd.to_datetime(old.date)
    joined=old.merge(base,on=["date","corridor"],validate="one_to_one")
    max_error=float((joined.probability-joined.base_probability).abs().max())
    if max_error>1e-10 or not np.array_equal(joined.candidate_signal,joined.base_candidate):raise ValueError("V2 reproduction failed")
    (OUT/"parity.json").write_text(json.dumps(dict(rows=len(joined),maximum_probability_error=max_error,candidate_mask_exact=True,bootstrap_reps=REPS,configurations=len(preds),status="PASS"),indent=2))
    print(summary[summary.track.eq("development_2023_2025")].sort_values("brier").head(10).to_string(index=False))
    print("Prediction parity",max_error,"; trials",len(preds))


if __name__=="__main__":main()
