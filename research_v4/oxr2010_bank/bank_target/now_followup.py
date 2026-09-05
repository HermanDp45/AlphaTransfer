"""Post-readout fixed NOW-event control on the same archived bank observations."""
from __future__ import annotations
import os,sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,time
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.bank_target import experiment as e
HERE=e.HERE;OUT=e.OUT/'now'

def now_panel(panel):
 p=panel.copy();ticks=np.rint(p.quote_kzt_per_rub*100).astype('int64');future=pd.concat({str(n):ticks.shift(-n) for n in range(1,6)},axis=1)
 p['future_min_kzt_per_rub']=future.min(axis=1,skipna=False)/100
 p['target']=(ticks<=future.min(axis=1,skipna=False)).astype(float).where(panel.target.notna())
 return p

def fit(tr,va,features,arm):
 bundle={'arm':arm,'features':features,'prior_probability':float(va.target.mean())}
 if arm in ('persistence','prior_mean'):return bundle
 model=e.make_model().fit(tr[features],tr.target.astype(int));raw=model.predict_proba(va[features])[:,1]
 cal=LogisticRegression(C=1.,solver='lbfgs',max_iter=500).fit(logit(raw.clip(1e-6,1-1e-6)).reshape(-1,1),va.target.astype(int))
 bundle.update(classifier=model,calibrator=cal);return bundle

def score(bundle,p):
 if bundle['arm'] in ('persistence','prior_mean'):return np.full(len(p),bundle['prior_probability']),np.full(len(p),bundle['prior_probability'])
 raw=bundle['classifier'].predict_proba(p[bundle['features']])[:,1];prob=bundle['calibrator'].predict_proba(logit(raw.clip(1e-6,1-1e-6)).reshape(-1,1))[:,1]
 return prob,raw

def metric(g):
 return {'rows':len(g),'dates':g.date.nunique(),'positive_share':float(g.target.mean()),'brier':float(np.mean((g.probability-g.target)**2)),'raw_brier':float(np.mean((g.raw_probability-g.target)**2)),'log_loss':float(log_loss(g.target,g.probability,labels=[0,1])),'auc':float(roc_auc_score(g.target,g.probability))}

def summarize(pred,aggregate=False):
 keys=['arm'] if aggregate else ['cutoff','arm'];rows=[]
 for key,g in pred.groupby(keys):
  k=key if isinstance(key,tuple) else (key,);rows.append({**dict(zip(keys,k)),**metric(g)})
 out=pd.DataFrame(rows)
 if aggregate:out['brier_skill_vs_past_prevalence']=1-out.brier/out.loc[out.arm.eq('persistence'),'brier'].iloc[0]
 else:
  base=out[out.arm.eq('persistence')].set_index('cutoff').brier;out['brier_skill_vs_past_prevalence']=1-out.brier/out.cutoff.map(base)
 return out

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);p,groups,_=e.build_panel();p=now_panel(p);receipts=[];outputs=[]
 with threadpool_limits(limits=1):
  for cutoff in ('2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'):
   tr,va,te=e.split(p,cutoff,str(int(cutoff[:4])+1)+'-01-01')
   for arm in e.ARMS:
    bundle=fit(tr,va,groups.get(arm,[]),arm);bundle['cutoff']=cutoff;prob,raw=score(bundle,te)
    q=te[['date','quote_date','decision_at','label_available_at','quote_ordinal','quote_kzt_per_rub','future_min_kzt_per_rub','target']].copy();q['probability']=prob;q['raw_probability']=raw;q['arm']=arm;q['cutoff']=cutoff;outputs.append(q)
    ckpt=OUT/'checkpoints'/f'{arm}_{cutoff}.pkl';ckpt.write_bytes(pickle.dumps(bundle));receipts.append({'arm':arm,'cutoff':cutoff,'features':groups.get(arm,[]),'prior_probability':bundle['prior_probability'],'train_rows':len(tr),'calibration_rows':len(va),'train_last_label':str(tr.label_available_at.max()),'calibration_last_label':str(va.label_available_at.max()),'checkpoint':str(ckpt.relative_to(HERE)),'checkpoint_sha256':e.sha(ckpt),'train_fingerprint':e.frame_fingerprint(tr[groups.get(arm,[])+['target']]),'calibration_fingerprint':e.frame_fingerprint(va[groups.get(arm,[])+['target']])})
   if cutoff=='2025-01-01':e.save(OUT/'development_completion.json',{'all_development_fits_before_2026_fits':True,'no_model_selection':True,'addendum_sha256':e.sha(HERE/'NOW_ADDENDUM.md')})
 allpred=pd.concat(outputs,ignore_index=True);allpred.to_csv(OUT/'predictions.csv.gz',index=False);development=allpred[allpred.date<'2026-01-01'];summarize(development,True).to_csv(OUT/'development_summary.csv',index=False);summarize(allpred).to_csv(OUT/'metrics_by_cutoff.csv',index=False);summarize(allpred[allpred.date.ge('2026-03-05')&allpred.cutoff.str.startswith('2026')]).to_csv(OUT/'common_march5_summary.csv',index=False)
 e.save(OUT/'model_receipts.json',receipts);e.save(OUT/'manifest.json',{'code_sha256':e.sha(Path(__file__)),'primary_code_sha256':e.sha(HERE/'experiment.py'),'addendum_sha256':e.sha(HERE/'NOW_ADDENDUM.md'),'source_manifest_sha256':e.sha(e.OUT/'manifest.json'),'predictions_sha256':e.sha(OUT/'predictions.csv.gz'),'seconds':time.monotonic()-started,'target':'q_anchor <= minimum next five actual BANK SELL RUB quotes; equality succeeds','post_primary_readout_followup':True,'no2026_selection':True})
 print(summarize(development,True).round(6).to_string(index=False),flush=True)
if __name__=='__main__':main()
