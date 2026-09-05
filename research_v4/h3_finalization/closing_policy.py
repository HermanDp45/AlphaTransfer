from pathlib import Path
import os,sys,json,shutil,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from final_solution.training import core_experiment as core
HERE=Path(__file__).resolve().parent;DEST=ROOT/'final_solution/tabm_h3'
def main():
 raw=pd.read_csv(HERE/'closing/raw_predictions.csv.gz',parse_dates=['date'],float_precision='round_trip')
 now=pd.read_csv(HERE/'predictions.csv.gz',parse_dates=['date'],float_precision='round_trip');now=now[(now.config_id=='tabm_kzt_fullhistory')&(now.policy=='rank80')]
 rows=[];heads=[];allpred=[];finalcal=None
 for kind,g in raw.groupby('fit_kind'):
  v=g[g.split=='validation'];cal=core.fit_platt_calibrator(v.raw_probability.to_numpy(),v.target)
  if kind=='final':finalcal=cal;continue
  t=g[g.split=='test'].copy();t['closing_probability']=core.apply_platt(cal,t.raw_probability.to_numpy());year=int(t.fold_test_year.iloc[0]);n=now[now.fold_test_year==year]
  q=t.merge(n[['date','candidate_signal']],on='date',validate='one_to_one');q['closing_annotation']=q.candidate_signal&q.closing_probability.ge(.5)&q.ret1.gt(0);s=q[q.closing_annotation]
  rows.append(dict(year=year,train_horizon=3,model='closing_hgb_kzt_fullhistory',policy='NOW_H3_rank80_AND_closing_p05_AND_ret1positive',now_signals=int(q.candidate_signal.sum()),closing_annotations=len(s),hit_rate=float(s.target.mean()),base_hit=float(q.target.mean()),lift=float(s.target.mean()/q.target.mean()),endpoint_delta_bps=float(s.closing_endpoint_bps.mean()-q.closing_endpoint_bps.mean()),extra_contacts=0))
  heads.append(dict(year=year,train_horizon=3,model='closing_hgb_kzt_fullhistory',rows=len(q),brier=float(np.mean((q.closing_probability-q.target)**2)),prior_constant_brier=float(np.mean((q.target-v.target.mean())**2)),calibration=cal.method))
  allpred.append(q)
 metrics=pd.DataFrame(rows);promote=bool((metrics.closing_annotations.ge(5)&metrics.lift.ge(1.3)&metrics.endpoint_delta_bps.gt(0)).all())
 metrics.to_csv(HERE/'closing_by_year.csv',index=False);pd.DataFrame(heads).to_csv(HERE/'closing_head_metrics.csv',index=False);pd.concat(allpred).to_csv(HERE/'closing_predictions.csv.gz',index=False)
 cp=DEST/'model/closing_h3.joblib';shutil.copy2(HERE/'closing/final/model.joblib',cp)
 bundle=json.loads((DEST/'bundle.json').read_text());bundle['closing']=dict(enabled=promote,scenario='CLOSING',train_horizon=3,target='R[t+3]>R[t]',model='model/closing_h3.joblib',model_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),features=bundle['features'],calibration=dict(method=finalcal.method,intercept=finalcal.intercept,slope=finalcal.slope),threshold=.5,requires_now=True,requires_positive_ret1=True,extra_contacts=False,status='retrospective_annotation_validated' if promote else 'diagnostic_only_failed_annual_annotation_gates',annual_gates='>=5 annotations,lift>=1.3,positiveendpointdelta ineachyear',annual_metrics=rows)
 bundle['metadata']['closing_status']=bundle['closing']['status'];(DEST/'bundle.json').write_text(json.dumps(bundle,indent=2,ensure_ascii=False)+'\n')
 (HERE/'closing_selection.json').write_text(json.dumps(bundle['closing'],indent=2)+'\n')
 print(metrics.to_string(index=False));print('CLOSING_H3 enabled',promote)
if __name__=='__main__':main()
