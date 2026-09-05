#!/usr/bin/env python3
"""Reproducible external-data ablation with the frozen AlphaTransfer evaluator.

Never fetches data, edits core code, tunes on test, or treats 2026 as holdout.
Run fetch.py first.  Restricted sources remain a separate counterfactual.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from final_solution.training import core_experiment as core
from final_solution.training import train_and_evaluate as old

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_simple(sid, family):
    raw = pd.read_csv(HERE / "raw" / f"{sid}.csv")
    datecol = "DATE" if family == "cboe" else "observation_date"
    valuecol = "CLOSE" if "CLOSE" in raw else raw.columns[-1]
    raw["source_date"] = pd.to_datetime(raw[datecol])
    return raw[["source_date", valuecol]].rename(columns={valuecol:sid}).dropna().sort_values("source_date")

def xml_curve(path):
    root = ET.fromstring(path.read_bytes())
    out = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        row = {}
        for item in entry.iter():
            name=item.tag.split("}")[-1]
            if name == "NEW_DATE": row["source_date"] = pd.Timestamp(item.text).normalize()
            elif name.startswith(("BC_", "TC_")):
                try: row[name] = float(item.text)
                except (TypeError, ValueError): pass
        if "source_date" in row: out.append(row)
    return pd.DataFrame(out)

def normalize():
    simple = {}
    for family, ids in {
        "cboe":["vix", "vix9d", "vvix", "ovx", "vxeem"],
        "fred":["t10yie", "t5yifr", "vixcls", "bamlh0a0hym2", "bamlemcbpioas"],
    }.items():
        for symbol in ids:
            sid = family + "_" + symbol
            simple[sid] = read_simple(sid, family)
            simple[sid].to_csv(HERE / "normalized" / f"{sid}.csv", index=False)
    nominal = pd.concat([xml_curve(HERE / "raw/treasury_nominal_2019.xml"), *[xml_curve(REPO / "final_solution/data/raw" / f"us_treasury_curve_{y}.xml") for y in range(2020,2027)]])
    real = pd.concat([xml_curve(HERE / "raw" / f"treasury_real_{y}.xml") for y in range(2019,2027)])
    curve = nominal.merge(real, on="source_date", how="inner").sort_values("source_date")
    curve["treasury_t10yie"] = curve["BC_10YEAR"] - curve["TC_10YEAR"]
    five = curve["BC_5YEAR"] - curve["TC_5YEAR"]
    # Published FRED definition, rounded to the source's two decimal precision.
    curve["treasury_t5yifr"] = (((1 + curve["treasury_t10yie"]/100)**10 / (1+five/100)**5)**0.2 -1)*100
    for symbol in ("t10yie", "t5yifr"):
        sid="treasury_" + symbol
        simple[sid] = curve[["source_date", sid]].copy()
        simple[sid][sid] = simple[sid][sid].round(2)
        simple[sid].to_csv(HERE / "normalized" / f"{sid}.csv", index=False)
    # Verify provider replication on common dates, before feature engineering.
    equivalence=[]
    for symbol in ("t10yie", "t5yifr", "vixcls"):
        left="fred_"+symbol
        right="cboe_vix" if symbol=="vixcls" else "treasury_"+symbol
        m=simple[left].merge(simple[right],on="source_date",how="inner").dropna()
        e=(m[left]-m[right]).abs()
        equivalence.append(dict(fred_series=left,primary_series=right,n=len(m),mean_absolute_difference=float(e.mean()),max_absolute_difference=float(e.max()),equal_share=float((e<1e-8).mean())))
    pd.DataFrame(equivalence).to_csv(HERE / "provider_equivalence.csv",index=False)
    gd=pd.read_excel(HERE / "raw/gpr_daily.xls")
    gd["source_date"]=pd.to_datetime(gd["date"])
    gd=gd[["source_date","GPRD","GPRD_ACT","GPRD_THREAT"]].sort_values("source_date")
    gm=pd.read_excel(HERE / "raw/gpr_monthly.xls")
    gm["source_date"]=pd.to_datetime(gm["month"])+pd.offsets.MonthEnd(0)
    gm=gm[["source_date","GPRC_RUS","GPRC_UKR","GPR"]].sort_values("source_date")
    gd.to_csv(HERE / "normalized/gpr_daily.csv",index=False)
    gm.to_csv(HERE / "normalized/gpr_monthly.csv",index=False)
    return simple, gd, gm

def build_panel():
    panel=old.build_feature_panel(REPO,REPO / "final_solution/data/normalized",core,old.PROFILES["primary"])
    simple,gd,gm=normalize()
    groups={}
    coverage=[]
    for stress in (False,True):
        suffix="_stale" if stress else ""
        for family in ("cboe","fred","treasury","credit"):
            ids={"cboe":["cboe_vix","cboe_vix9d","cboe_vvix","cboe_ovx","cboe_vxeem"],
                 "fred":["fred_t10yie","fred_t5yifr"],
                 "treasury":["treasury_t10yie","treasury_t5yifr"],
                 "credit":["fred_bamlh0a0hym2","fred_bamlemcbpioas"]}[family]
            features=None
            names=[]
            for sid in ids:
                frame=simple[sid].copy()
                prefix=sid+suffix
                value=frame[sid]
                f=frame[["source_date"]].copy()
                if family=="cboe":
                    f[prefix+"_loglevel"]=np.log(value)
                    f[prefix+"_logret5"]=np.log(value).diff(5)
                else:
                    f[prefix+"_level"]=value
                    f[prefix+"_chg5"]=value.diff(5)
                    f[prefix+"_chg20"]=value.diff(20)
                names += [c for c in f if c!="source_date"]
                features=f if features is None else features.merge(f,on="source_date",how="outer")
            if family=="cboe":
                for title,left,right in [("short_term_slope","vix9d","vix"),("em_relative","vxeem","vix")]:
                    name="cboe_"+title+suffix
                    features[name]=features["cboe_"+left+suffix+"_loglevel"]-features["cboe_"+right+suffix+"_loglevel"]
                    names.append(name)
            lag=7 if stress else 2
            features=old.add_availability(features,lag)
            panel=old.asof_join(panel,features,"v3_"+family+suffix,14 if stress else 7)
            groups[family+suffix]=names
        # Daily GPR is published weekly. Fourteen days covers weekly publication,
        # timezones and ordinary holiday delays; 30-day sensitivity is stricter.
        f=gd[["source_date"]].copy()
        names=[]
        for raw in ("GPRD","GPRD_ACT","GPRD_THREAT"):
            prefix="gpr_"+raw.lower()+suffix
            value=np.log1p(gd[raw])
            f[prefix+"_mean7"]=value.rolling(7).mean()
            f[prefix+"_mean30"]=value.rolling(30).mean()
            f[prefix+"_accel7"]=f[prefix+"_mean7"].diff(7)
            names += [prefix+"_mean7",prefix+"_mean30",prefix+"_accel7"]
        panel=old.asof_join(panel,old.add_availability(f,30 if stress else 14),"v3_gpr_daily"+suffix,45)
        f=gm[["source_date"]].copy()
        for raw in ("GPRC_RUS","GPRC_UKR"):
            prefix="gpr_"+raw.lower()+suffix
            f[prefix+"_level"]=np.log1p(gm[raw])
            f[prefix+"_chg3"]=np.log1p(gm[raw]).diff(3)
            names += [prefix+"_level",prefix+"_chg3"]
        panel=old.asof_join(panel,old.add_availability(f,45 if stress else 15),"v3_gpr_monthly"+suffix,90)
        groups["gpr"+suffix]=names
    for family,names in groups.items():
        for year,frame in panel.groupby(panel.date.dt.year):
            coverage.append(dict(family=family,year=int(year),features=len(names),rows=len(frame),complete_row_share=float(frame[names].notna().all(axis=1).mean()),average_feature_coverage=float(frame[names].notna().mean().mean())))
    pd.DataFrame(coverage).to_csv(HERE / "feature_coverage.csv",index=False)
    panel.to_parquet(HERE / "feature_panel.parquet",index=False)
    return panel,groups

def paired_audit(pred, pairs, reps=10000):
    rng=np.random.default_rng(20260904)
    rows=[]
    for candidate,baseline in pairs:
        keys=["date","corridor","fold_test_year"]
        a=pred[pred.config_id.eq(candidate)]
        b=pred[pred.config_id.eq(baseline)]
        m=a.merge(b,on=keys,suffixes=("_new","_base"),validate="one_to_one")
        if m.empty:continue
        assert (m.target_new==m.target_base).all()
        m["delta"]=(m.probability_new-m.target_new)**2-(m.probability_base-m.target_base)**2
        by_date=m.groupby(["date","fold_test_year"],as_index=False).delta.mean()
        by_date["month"]=by_date.date.dt.to_period("M")
        blocks=by_date.groupby(["fold_test_year","month"]).delta.agg(["sum","count"]).reset_index()
        boot_num=np.zeros(reps);boot_den=np.zeros(reps)
        for _,g in blocks.groupby("fold_test_year"):
            idx=rng.integers(0,len(g),size=(reps,len(g)))
            boot_num+=g["sum"].to_numpy()[idx].sum(axis=1)
            boot_den+=g["count"].to_numpy()[idx].sum(axis=1)
        lo,hi=np.quantile(boot_num/boot_den,[.025,.975])
        baseb=np.mean((m.probability_base-m.target_base)**2)
        yearly=m.groupby("fold_test_year").delta.mean()
        cells=m.groupby(["fold_test_year","corridor"]).delta.mean()
        rows.append(dict(candidate=candidate,baseline=baseline,paired_rows=len(m),decision_dates=m.date.nunique(),month_blocks=len(blocks),brier_delta=float(m.delta.mean()),relative_brier_improvement=float(-m.delta.mean()/baseb),ci95_low=float(lo),ci95_high=float(hi),improved_years=int((yearly<0).sum()),years=len(yearly),improved_cells=int((cells<0).sum()),cells=len(cells),bootstrap_reps=reps,interpretation="retrospective exploratory; no multiplicity-adjusted confirmation"))
    return pd.DataFrame(rows)

def metrics(pred,folds):
    out=old.merge_predictive_and_policy(core,folds,pred)
    raw=[]
    for key,g in pred.groupby("config_id"):
        raw.append(dict(config_id=key,raw_brier=float(np.mean((g.raw_probability-g.target)**2))))
    return out.merge(pd.DataFrame(raw),on="config_id")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","primary","sensitivity","credit","replica","sensitivity_replica"],default="all")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        panel,ext=build_panel()
        groups=old.feature_groups(core)
        base=groups["base"]
        basis=groups["plus_cnyrub_basis"]
        core.FEATURE_GROUPS.update(groups)
        specs=[];pairs=[]
        def add(cid,features,baseid):
            core.FEATURE_GROUPS[cid]=features
            specs.append(core.Experiment(cid,"hist_gradient_boosting",cid))
            if baseid:pairs.append((cid,baseid))
        for baseid,features in [("hgb_base",base),("hgb_plus_cnyrub_basis",basis)]:
            add(baseid,features,None)
            for family in ("fred","cboe","gpr"):
                add(baseid+"__"+family,features+ext[family],baseid)
        add("hgb_plus_cnyrub_basis__treasury",basis+ext["treasury"],"hgb_plus_cnyrub_basis")
        add("hgb_plus_cnyrub_basis__all_new",basis+ext["fred"]+ext["cboe"]+ext["gpr"],"hgb_plus_cnyrub_basis")
        if args.stage in ("all","primary","replica"):
            targeted=core.add_target(panel,5)
            foldrows=[];preds=[]
            run_specs=specs
            if args.stage=="replica":
                run_specs=[s for s in specs if s.config_id=="hgb_plus_cnyrub_basis__treasury"]
                prior_p=pd.read_csv(HERE / "predictions.csv",parse_dates=["date"])
                prior_f=pd.read_csv(HERE / "fold_corridor_metrics.csv")
                preds=[prior_p[~prior_p.config_id.eq(run_specs[0].config_id)]]
                foldrows=prior_f[~prior_f.config_id.eq(run_specs[0].config_id)].to_dict("records")
            for spec in run_specs:
                for year in (2023,2024,2025,2026):
                    print("primary",spec.config_id,year,flush=True)
                    rows,p,_=core.run_fold(spec,5,year,targeted)
                    p["model_kind"]=spec.model_kind;p["feature_set"]=spec.feature_set
                    foldrows.extend(rows);preds.append(p)
                    pd.concat(preds).to_csv(HERE / "predictions.partial.csv",index=False)
            pred=pd.concat(preds,ignore_index=True);folds=pd.DataFrame(foldrows)
            pred.to_csv(HERE / "predictions.csv",index=False);folds.to_csv(HERE / "fold_corridor_metrics.csv",index=False)
            for label,years in [("development",[2023,2024,2025]),("diagnostic_2026",[2026])]:
                p=pred[pred.fold_test_year.isin(years)];f=folds[folds.fold_test_year.isin(years)]
                metrics(p,f).to_csv(HERE / (label+"_metrics.csv"),index=False)
                paired_audit(p,pairs).to_csv(HERE / (label+"_paired_audit.csv"),index=False)
            (HERE / "predictions.partial.csv").unlink(missing_ok=True)
        if args.stage in ("all","sensitivity","sensitivity_replica"):
            foldrows=[];preds=[]
            families=("fred","cboe","gpr","treasury")
            if args.stage=="sensitivity_replica":
                families=("treasury",)
                prior=pd.read_csv(HERE / "sensitivity_predictions.csv",parse_dates=["date"])
                preds=[prior[~prior.config_id.eq("hgb_plus_cnyrub_basis__treasury_stale")]]
                # Retain existing policy aggregates. New Treasury metrics are
                # appended below without recomputing unrelated models.
            for family in families:
                cid="hgb_plus_cnyrub_basis__"+family+"_stale"
                core.FEATURE_GROUPS[cid]=basis+ext[family+"_stale"]
                spec=core.Experiment(cid,"hist_gradient_boosting",cid)
                for year in (2023,2024,2025,2026):
                    print("stale",family,year,flush=True)
                    rows,p,_=core.run_fold(spec,5,year,core.add_target(panel,5))
                    p["model_kind"]=spec.model_kind;p["feature_set"]=spec.feature_set
                    foldrows.extend(rows);preds.append(p)
            pred=pd.concat(preds,ignore_index=True);folds=pd.DataFrame(foldrows)
            pred.to_csv(HERE / "sensitivity_predictions.csv",index=False)
            for label,years in [("development",[2023,2024,2025]),("diagnostic_2026",[2026])]:
                fresh=pred[pred.fold_test_year.isin(years)&pred.config_id.isin(folds.config_id)]
                current=metrics(fresh,folds[folds.fold_test_year.isin(years)])
                path=HERE / (label+"_stale_metrics.csv")
                if args.stage=="sensitivity_replica":
                    prior=pd.read_csv(path)
                    current=pd.concat([prior[~prior.config_id.isin(current.config_id)],current],ignore_index=True)
                current.to_csv(path,index=False)
                bases=pd.read_csv(HERE / "predictions.csv",parse_dates=["date"])
                combined=pd.concat([bases,pred],ignore_index=True)
                combined=combined[combined.fold_test_year.isin(years)]
                comparisons=[("hgb_plus_cnyrub_basis__"+f+"_stale","hgb_plus_cnyrub_basis") for f in ("fred","cboe","gpr","treasury")]
                paired_audit(combined,comparisons).to_csv(HERE / (label+"_stale_paired_audit.csv"),index=False)
        if args.stage in ("all","credit"):
            # ICE free history now begins Sep-2023. Never zero-fill an absent
            # pre-2023 history and call it a full development ablation.
            core.TRAIN_WINDOW_YEARS=1
            preds=[];foldrows=[]
            for addon in (False,True):
                cid="credit_matched_2026"+("__ice" if addon else "__basis")
                core.FEATURE_GROUPS[cid]=basis+(ext["credit"] if addon else [])
                spec=core.Experiment(cid,"hist_gradient_boosting",cid)
                print("credit",cid,2026,flush=True)
                rows,p,_=core.run_fold(spec,5,2026,core.add_target(panel,5))
                p["model_kind"]=spec.model_kind;p["feature_set"]=spec.feature_set
                preds.append(p);foldrows.extend(rows)
            p=pd.concat(preds,ignore_index=True);f=pd.DataFrame(foldrows)
            p.to_csv(HERE / "credit_matched_2026_predictions.csv",index=False)
            metrics(p,f).to_csv(HERE / "credit_matched_2026_metrics.csv",index=False)
            paired_audit(p,[("credit_matched_2026__ice","credit_matched_2026__basis")]).to_csv(HERE / "credit_matched_2026_paired_audit.csv",index=False)
            core.TRAIN_WINDOW_YEARS=2
        manifest=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),stage=args.stage,python=sys.version,train_years=2,validation_years=1,test_years=[2023,2024,2025],diagnostic_year=2026,primary_horizon=5,calibration_and_policy="unmodified core.run_fold",decision_date_status="CBR effective-date proxy, not verified publication timestamp",latest_snapshot_sources=True,source_timing={"US_market":"observation +2 calendar days; stale +7", "daily_GPR":"+14 days; stale +30", "monthly_GPR":"month end +15 days; stale +45"},credit="2026 diagnostic matched train2024 validation2025",core_hash=digest(REPO / "final_solution/training/core_experiment.py"),old_training_hash=digest(REPO / "final_solution/training/train_and_evaluate.py"),script_hash=digest(__file__),source_classification="FRED/CBOE/ICE counterfactual research only; Treasury primary replicates separately",files={str(p.relative_to(HERE)):digest(p) for directory in ("raw","normalized") for p in sorted((HERE/directory).glob("*")) if p.is_file()})
        (HERE / "experiment_manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
