"""Finite final sprint: recency, source dropout, shrinking and feature groups."""
from pathlib import Path
import os,sys,json,pickle,time,warnings
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
from research_v4.final_sprint.common import WeightedAdapted,POLICIES,fit_policy,apply_policy,metrics
HERE=Path(__file__).resolve().parent;OUT=HERE/'hgb';OUT.mkdir(exist_ok=True)

def specs():
    rows=[]
    def add(m=120,cal=12,decay=0,features='bank_treasury',augment=False,adapt=True,depth=2):
        name=f'hgb_{m}m_c{cal}_d{decay}_{features}'+('_aug' if augment else '')+('' if adapt else '_pooled')+('' if depth==2 else '_stump')
        rows.append(dict(name=name,months=m,cal=cal,decay=decay,features=features,augment=augment,adapt=adapt,depth=depth))
    for m in (12,24,36,60,120):
        for cal in (12,3):add(m,cal)
    for cal in (12,3):
        for decay in (1,3,6,12):add(120,cal,decay)
    for cal in (12,3):
        add(120,cal,features='bank_treasury_oxr')
        add(60,cal,augment=True)
        add(120,cal,augment=True)
        add(120,cal,features='compact')
        add(120,cal,features='bank_only')
        add(120,cal,adapt=False)
        add(120,cal,depth=1)
    return rows

def split(p,s,cutoff):
    calstart=cutoff-pd.DateOffset(months=s['cal']);trainstart=calstart-pd.DateOffset(months=s['months'])
    tr=p[p.date.ge(trainstart)&p.date.lt(calstart)&p.label_available_date.lt(calstart)&p.target.notna()].copy()
    va=p[p.date.ge(calstart)&p.date.lt(cutoff)&p.label_available_date.lt(cutoff)&p.target.notna()&p.corridor.eq('KZT')].copy()
    hist=p[p.date.ge(calstart)&p.date.lt(cutoff)&p.corridor.eq('KZT')].copy()
    te=p[p.date.ge(cutoff)&p.date.lt(pd.Timestamp(cutoff.year+1,1,1))&p.target.notna()&p.corridor.eq('KZT')].copy()
    # Same annual evaluation grid as sealed V3; exclude last five target rows of complete historical years.
    if cutoff.year<2026:te=e.core.purge_tail(te,5)
    assert tr.label_available_date.max()<calstart and va.label_available_date.max()<cutoff
    return tr,va,hist,te

def run(views,bankcols,s,cutoff):
    cutoff=pd.Timestamp(cutoff);p=views['2010-01-01',24,1]
    feat=e.oxr.BASE+bankcols+TREASURY_LAG7_FEATURES
    if s['features']=='bank_treasury_oxr':feat+=e.oxr.BASIS+e.oxr.COVER
    if s['features']=='bank_only':feat=e.oxr.BASE+bankcols
    if s['features']=='compact':feat=['ret1','ret5','ret20','pr20','pr60','pr252','vol20',e.oxr.BASE[-1]]+bankcols[:2]+[TREASURY_LAG7_FEATURES[0],TREASURY_LAG7_FEATURES[3]]
    tr,va,history,te=split(p,s,cutoff);original=len(tr)
    if s['augment']:
        tr=pd.concat([tr,views['2010-01-01',24,2].loc[tr.index]],ignore_index=True)
    weights=np.ones(len(tr))
    if s['decay']:
        age=(tr.date.max()-tr.date).dt.days.to_numpy()/30.4375
        weights=np.exp2(-age/s['decay']);weights/=weights.mean()
    model=e.new_model(feat);model.named_steps['classifier'].set_params(max_depth=s['depth'])
    model=WeightedAdapted(model,s['adapt']).fit(tr[feat+['corridor']],tr.target.astype(int),weights)
    if s['adapt']:
        x=va[feat+['corridor']];pooled=model.pooled.predict_proba(x)[:,1];adapted=model.predict_proba(x)[:,1]
        model.weight=min((0.,.25,.5,.75,1.),key=lambda w:np.mean((w*adapted+(1-w)*pooled-va.target)**2))
    raw=model.predict_proba(va[feat+['corridor']])[:,1];cal=e.core.fit_platt_calibrator(raw,va.target)
    for z in (va,history):
        z['raw_probability']=model.predict_proba(z[feat+['corridor']])[:,1]
        z['probability']=e.core.apply_platt(cal,z.raw_probability.to_numpy())
    policies={name:fit_policy(va,history,name) for name in POLICIES}
    predictions=[];summaries=[]
    modes={'normal':(24,1),'bank_delayed':(24,2)}
    if s['features']=='bank_treasury_oxr':modes.update(oxr_delayed=(48,1),both_delayed=(48,2))
    for mode,(delay,lag) in modes.items():
        z=e.stress_view(views,dict(since='2010-01-01'),cutoff,delay,lag).loc[te.index].copy()
        z['raw_probability']=model.predict_proba(z[feat+['corridor']])[:,1]
        z['probability']=e.core.apply_platt(cal,z.raw_probability.to_numpy())
        for name,policy in policies.items():
            q=apply_policy(z,policy);q['mode']=mode;q['config_id']=s['name'];q['cutoff']=str(cutoff.date());q['fold_test_year']=cutoff.year
            summaries.append(dict(config_id=s['name'],cutoff=str(cutoff.date()),mode=mode,policy=name,**metrics(q)))
            predictions.append(q[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date','raw_probability','probability','candidate_signal','policy','mode','config_id','cutoff','fold_test_year','pr60','ret1','rub_per_unit']])
    name=s['name']+'_'+str(cutoff.date())
    pd.concat(predictions).to_csv(OUT/(name+'.csv.gz'),index=False)
    histories=[]
    for splitname,z in [('validation',va),('history',history)]:
        z=z[['date','corridor','session_ordinal','target','forward_bps','label_available_date','raw_probability','probability']].copy()
        z.loc[z.label_available_date.ge(cutoff)|z.label_available_date.isna(),['target','forward_bps']]=np.nan
        z['split']=splitname;z['config_id']=s['name'];z['cutoff']=str(cutoff.date());histories.append(z)
    pd.concat(histories).to_csv(OUT/(name+'_calibration.csv.gz'),index=False)
    ckpt=OUT/(name+'.pkl');ckpt.write_bytes(pickle.dumps(dict(model=model,calibrator=cal,features=feat,policies=policies,spec=s,cutoff=str(cutoff.date()))))
    e.save(OUT/(name+'.json'),dict(spec=s,features=feat,train_rows=original,fit_rows=len(tr),train_min=tr.date.min(),train_max=tr.date.max(),latest_train_label=tr.label_available_date.max(),validation_min=va.date.min(),validation_max=va.date.max(),latest_validation_label=va.label_available_date.max(),test_rows=len(te),residual_weight=model.weight,weight_ess=float(weights.sum()**2/(weights**2).sum()),checkpoint_sha256=e.sha(ckpt),policies=policies))
    print(name,'done',flush=True)
    return summaries

def main():
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,bankcols=e.build_views();views={k:augment_panel(v) for k,v in views.items()}
        with (HERE/'views.pkl').open('wb') as f:pickle.dump((views,bankcols),f)
        summaries=[]
        for year in (2025,2026):
            for s in specs():summaries+=run(views,bankcols,s,f'{year}-01-01')
            pd.DataFrame(summaries).to_csv(OUT/'metrics.csv',index=False)
        e.save(OUT/'completion.json',dict(fits=len(specs())*2,configs=len(specs()),status='complete'))
if __name__=='__main__':main()
