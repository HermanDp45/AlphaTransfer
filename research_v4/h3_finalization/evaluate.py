from pathlib import Path
import os,sys,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.evaluate import metric,ranked,policy_fit,apply,NAMES,core
from research_v4.continuation.oxr.assess import paired
HERE=Path(__file__).resolve().parent

def main():
 parts=[]
 for folder,isbase in [(ROOT/'research_v4/robust_selection/tabm',True),(HERE/'long_history',False)]:
  for filename in ('raw_predictions.csv.gz','warmup.csv.gz'):
   z=pd.read_csv(folder/filename,parse_dates=['date','label_available_date'],float_precision='round_trip')
   if isbase:z=z[(z.config_id=='tabm_kzt')&(z.train_horizon==3)].copy();z['config_id']='tabm_kzt_120m'
   parts.append(z)
 raw=pd.concat(parts,ignore_index=True);out=[];pols=[];cals=[]
 for (cid,year),g in raw.groupby(['config_id','fold_test_year']):
  v,h,t,w=[g[g.split==s].copy() for s in ('validation','history','test','warmup')]
  assert len(w)==63 and v.label_available_date.lt(pd.Timestamp(year,1,1)).all()
  cal=core.fit_platt_calibrator(v.raw_probability.to_numpy(),v.target)
  for z in (v,h,t,w):z['probability']=core.apply_platt(cal,z.raw_probability.to_numpy())
  ref=ranked(pd.concat([w,h,t]).sort_values('date')).set_index('date').rank_score
  for z in (v,h,t):z['rank_score']=z.date.map(ref)
  cals.append(dict(config_id=cid,year=int(year),method=cal.method,intercept=cal.intercept,slope=cal.slope))
  for name in NAMES:
   pol=policy_fit(v,h,name);q=apply(t,pol);out.append(q);pols.append(dict(config_id=cid,year=int(year),**pol))
 pred=pd.concat(out,ignore_index=True);pred.to_csv(HERE/'predictions.csv.gz',index=False)
 for name,obj in [('policies.json',pols),('calibration.json',cals)]: (HERE/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
 rows=[];agg=[];ranks=[];cis=[]
 for (cid,policy),g in pred.groupby(['config_id','policy']):
  ys=[]
  for year,z in g.groupby('fold_test_year'):
   row=dict(config_id=cid,policy=policy,year=year,train_horizon=3,evaluation_horizon=3,evaluation_scope='KZT',**metric(z));rows.append(row);ys.append(row)
  agg.append(dict(config_id=cid,policy=policy,period='2024-2026',train_horizon=3,evaluation_horizon=3,evaluation_scope='KZT',**metric(g)))
  cover=np.array([x['week_coverage'] for x in ys]);lift=np.array([x['lift'] for x in ys]);util=np.array([x['forward_delta_bps'] for x in ys]);qual=bool(np.all(cover>=.8)&np.all(lift>=1.3)&np.all(util>0))
  ranks.append(dict(config_id=cid,policy=policy,qualified=qual,preferred90=qual and bool(np.all(cover>=.9)),gate_shortfall=float(np.maximum(0,(.8-cover)/.8).mean()+np.maximum(0,(1.3-np.nan_to_num(lift))/1.3).mean()+np.mean(util<=0)),min_coverage=float(cover.min()),min_lift=float(np.nan_to_num(lift).min()),min_utility=float(util.min()),brier=float(np.mean([x['brier'] for x in ys]))))
 ranked_rows=sorted(ranks,key=lambda x:(not x['qualified'],not x['preferred90'],x['gate_shortfall'] if not x['qualified'] else 0,-x['min_lift'],-x['min_utility'],x['brier']))
 for i,r in enumerate(ranked_rows):r['rank']=i+1
 for policy in NAMES:
  a=pred[(pred.config_id=='tabm_kzt_120m')&(pred.policy==policy)];b=pred[(pred.config_id=='tabm_kzt_fullhistory')&(pred.policy==policy)]
  cis.append(dict(baseline='tabm_kzt_120m',candidate='tabm_kzt_fullhistory',policy=policy,train_horizon=3,evaluation_scope='KZT',**paired(a,b)))
 selection=dict(chosen=ranked_rows[0],best_per_history={cid:next(x for x in ranked_rows if x['config_id']==cid) for cid in pred.config_id.unique()},qualification='qualified' if ranked_rows[0]['qualified'] else 'fallback_no_qualifying_H3',period='2024-2026_retrospective',no_H5_substitution=True)
 for name,rs in [('by_year.csv',rows),('summary.csv',agg),('ranking.csv',ranked_rows),('paired_intervals.csv',cis)]:pd.DataFrame(rs).to_csv(HERE/name,index=False)
 (HERE/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 print(json.dumps(selection,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
