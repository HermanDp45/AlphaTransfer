"""Separate frozen-H5 rescoring, matched-date H3/H5, and optional in-sample warmup."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,hashlib
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.v3 import experiment as e
HERE=e.HERE

def point(q,mask):
    q=q.copy();q['selected']=mask
    bases=q.groupby(['fold_test_year','corridor']).agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
    s=q[q.selected].join(bases,on=['fold_test_year','corridor'])
    coverage=[];legacy=[]
    for _,g in q.groupby(['fold_test_year','corridor']):
        counts=g.groupby(g.date.dt.to_period('W-SUN')).selected.sum();coverage.append(float(counts.between(1,2).mean()))
        legacy.append(e.core.cadence_diagnostics(g,g.index[g.selected])['weeks_with_1_to_2_signals_share'])
    return dict(rows=len(q),dates=q.date.nunique(),contacts=len(s),hits=int(s.target.sum()),hit_rate=float(s.target.mean()) if len(s) else None,lift_standardized=float(s.target.sum()/s.base_hit.sum()) if len(s) else None,lift_unstandardized=float(s.target.mean()/q.target.mean()) if len(s) else None,forward_delta_bps=float((s.forward_bps-s.base_forward).mean()) if len(s) else None,forward_delta_unstandardized_bps=float(s.forward_bps.mean()-q.forward_bps.mean()) if len(s) else None,mean_all_observed_week_coverage=float(np.mean(coverage)),mean_legacy_full_week_coverage=float(np.mean(legacy)))

def warmup(raw):
    rows=[];receipts=[]
    for r in json.loads((HERE/'model_receipts.json').read_text()):
        h=r['train_horizon'];year=r['year'];panel=e.target_panel(raw,h);tr=e.parts(panel,h,year)['train'];past=panel[panel.date.lt(pd.Timestamp(year-1,1,1))];days=sorted(past.date.unique())[-63:];q=past[past.date.isin(days)].copy();cp=HERE/r['checkpoint'];b=joblib.load(cp)
        rawp=b['model'].predict_proba(q[e.FEATURES+['corridor']])[:,1];p=e.core.apply_platt(b['calibrator'],rawp)
        out=q[e.KEEP].copy();out[e.OUTCOME]=np.nan;out['raw_probability']=rawp;out['probability']=p;out['original_v3_probability']=p
        out['config_id']='v3';out['train_horizon']=h;out['fold_test_year']=year;out['cutoff']=f'{year}-01-01';out['split']='warmup';out['in_sample_training_warmup']=q.index.isin(tr.index);out['calibration_scope']='pooled5_original_v3';rows.append(out)
        receipts.append(dict(train_horizon=h,year=year,rows=len(q),dates=q.date.nunique(),min_date=str(q.date.min()),max_date=str(q.date.max()),checkpoint=r['checkpoint'],checkpoint_sha256=e.sha(cp),same_frozen_model_and_prior_year_calibrator=True,all_outcomes_masked=True,in_sample_training_rows=int(q.index.isin(tr.index).sum()),purged_train_tail_rows=int((~q.index.isin(tr.index)).sum()),includes_last_purged_training_tail=True,warning='Last63 PANEL dates before calibration start, including horizon-purged training tail; many rows are in-sample. Same frozen model; solely optional policy state warmup. Not OOT predictions or matured-label evidence. Original V3 policy/parity does not consume this additional warmup.'))
    p=pd.concat(rows,ignore_index=True);p.to_csv(HERE/'warmup.csv.gz',index=False);e.save(HERE/'warmup_receipt.json',dict(rows=len(p),sha256=e.sha(HERE/'warmup.csv.gz'),parts=receipts));return p

def matched(pred):
    test=pred[pred.split.eq('test')];res=[]
    for year in (2024,2025,2026):
        parts={h:test[test.train_horizon.eq(h)&test.fold_test_year.eq(year)].set_index(['date','corridor']) for h in (3,5)}
        keys=parts[3].index.intersection(parts[5].index)
        for source in (3,5):
            for target in (3,5):
                policy_frame=parts[source].loc[keys];outcome_frame=parts[target].loc[keys].reset_index()
                for scope in ['all',*sorted(outcome_frame.corridor.unique())]:
                    selected_scope=np.ones(len(outcome_frame),bool) if scope=='all' else outcome_frame.corridor.eq(scope).to_numpy()
                    q=outcome_frame.loc[selected_scope].copy()
                    for col in ['candidate_signal','strict05_candidate_signal','signal','strict05_signal']:
                        mask=policy_frame[col].to_numpy()[selected_scope]
                        res.append(dict(train_horizon=source,evaluate_horizon=target,year=year,scope=scope,policy=col,interpretation='separately_trained_matched_date_evaluation' if source==target else 'frozen_policy_rescore_no_refit',**point(q,mask)))
    frame=pd.DataFrame(res);frame.to_csv(HERE/'matched_horizon_rescore.csv',index=False);return frame

def colleagues(raw):
    h5=pd.read_csv(ROOT/'research_v3/models/basis_train_120m_h5_predictions.csv.gz',parse_dates=['date']);h5=h5[h5.fold_test_year.le(2025)&h5.corridor.eq('KZT')].copy()
    p3=e.target_panel(raw,3).set_index(['date','corridor']);keys=pd.MultiIndex.from_frame(h5[['date','corridor']]);q=h5.copy();q[e.OUTCOME]=p3.loc[keys,e.OUTCOME].to_numpy()
    rows=[]
    # Reproduce the strict mask from the available historical H5 scores only.
    for threshold_kind in ['threshold_only','cooldown_replay_test_annual','cooldown_replay_test_continuous']:
        mask=pd.Series(False,index=h5.index)
        if threshold_kind=='threshold_only':mask=h5.probability.ge(.5)
        elif threshold_kind=='cooldown_replay_test_annual':
            for _,g in h5.groupby('fold_test_year'):
                ix=e.core.select_per_corridor_with_cooldown(g,g.probability.to_numpy(),.5);mask.loc[ix]=True
        else:
            ix=e.core.select_per_corridor_with_cooldown(h5,h5.probability.to_numpy(),.5);mask.loc[ix]=True
        rows.append(dict(method=threshold_kind,train_horizon=5,evaluate_horizon=3,years='2023–2025',scope='KZT',interpretation='H5 trained probabilities/signals evaluated against H3 outcomes, no H3 refit',**point(q,mask.to_numpy())))
    out=pd.DataFrame(rows);out.to_csv(HERE/'colleague_h3_reproduction.csv',index=False)
    e.save(HERE/'colleague_h3_reproduction.json',{'reported_target':{'rows':727,'contacts':60,'hits':39,'hit_rate':.65,'lift_standardized':1.921938637,'forward_delta_bps':81.91052},'reproductions':rows,'interpretation':'Identical frequency is mechanical when the same H5 mask is retained. The horizon of the trained probability remains H5. This is valid sensitivity analysis if labelled as such, not a separately trained H3 model.'})
    return out

def main():
    raw=pd.read_pickle(e.SOURCE);pred=pd.read_csv(HERE/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'])
    with threadpool_limits(limits=1):
        w=warmup(raw);m=matched(pred);c=colleagues(raw)
    print('WARMUP_READY',len(w));print(c.to_string(index=False))
if __name__=='__main__':main()
