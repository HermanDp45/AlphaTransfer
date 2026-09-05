"""Past-only Kazakhstan adaptation with exact V3 KZT comparators.

Newton residual adaptation freezes the pooled HGB and adds 40 shallow trees
fitted only on KZT. Pooled logits retain their float64 input precision: sklearn
GradientBoostingClassifier would cast its init estimator's input to float32.
"""
from __future__ import annotations
from pathlib import Path
import sys, os, json, hashlib, pickle, copy, time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_v3.models import experiment as old
from final_solution.training import core_experiment as core

HERE=Path(__file__).resolve().parent
FEATURES=old.BASE_FEATURES+[old.BASIS]
KZT='KZT'

class Adapted:
    def __init__(self, pooled, strategy):self.pooled,self.strategy=pooled,strategy
    def fit(self,x,y):
        self.pooled.fit(x,y)
        if self.strategy in ('residual','residual_shrink'):
            mask=x.corridor.eq(KZT)
            z=self.pooled.named_steps['preprocessor'].transform(x.loc[mask])
            labels=y.loc[mask].to_numpy(float)
            margin=logit(self.pooled.predict_proba(x.loc[mask])[:,1].clip(1e-7,1-1e-7))
            self.trees=[];self.learning_rate=.025
            for step in range(40):
                prob=expit(margin);hessian=(prob*(1-prob)).clip(1e-5)
                residual=(labels-prob)/hessian
                tree=DecisionTreeRegressor(max_depth=1,min_samples_leaf=60,random_state=core.SEED+step)
                tree.fit(z,residual,sample_weight=hessian)
                margin+=self.learning_rate*tree.predict(z)
                self.trees.append(tree)
        return self
    def predict_proba(self,x):
        if self.strategy in ('residual','residual_shrink'):
            original=self.pooled.predict_proba(x)
            z=self.pooled.named_steps['preprocessor'].transform(x)
            correction=sum(self.learning_rate*tree.predict(z) for tree in self.trees)
            score=expit(logit(original[:,1].clip(1e-7,1-1e-7))+correction)
            adapted=np.column_stack([1-score,score])
            weight=getattr(self,'weight',1.)
            return weight*adapted+(1-weight)*original
        return self.pooled.predict_proba(x)

# Stable pickle namespace for both direct CLI execution and library imports.
if __name__ in ('__main__','experiment'):
    sys.modules['research_v4.kazakhstan.experiment']=sys.modules[__name__]
    Adapted.__module__='research_v4.kazakhstan.experiment'

def evaluate(months,strategy,*,panel=None,extra_features=(),tag=''):
    """KZT-only OOT output, pooling only during train where applicable."""
    source=ROOT/'research_v3/models'/('panel_extended.pkl' if months==120 else 'panel_v2.pkl')
    p=pd.read_pickle(source) if panel is None else panel.copy()
    features=FEATURES+list(extra_features)
    p=core.add_target(p,5)
    name=f'kzt_{strategy}_{months}m'+tag
    outputs=[];all_cells=[];fits=[]
    for year in (2023,2024,2025,2026):
        spec=old.Spec(name,months=months,extended=months==120)
        tr,va,te=old.temporal_split(p,5,pd.Timestamp(year,1,1),pd.Timestamp(year+1,1,1),spec)
        train=tr[tr.corridor.eq(KZT)] if strategy=='only' else tr
        val=va[va.corridor.eq(KZT)].copy();test=te[te.corridor.eq(KZT)].copy()
        model=core.make_model('hist_gradient_boosting',features)
        model.named_steps['classifier'].set_params(early_stopping=False)
        model=Adapted(model,strategy)
        start=time.monotonic();model.fit(train[features+['corridor']],train.target.astype(int))
        if strategy=='residual_shrink':
            vx=val[features+['corridor']]
            pooled=model.pooled.predict_proba(vx)[:,1]
            adapted=model.predict_proba(vx)[:,1]
            model.weight=min((0.,.25,.5,.75,1.),key=lambda w:float(np.mean((w*adapted+(1-w)*pooled-val.target)**2)))
        # core.run_fold will fit this already fitted object through a no-op shell.
        class Ready:
            def fit(self,x,y):return self
            def predict_proba(self,x):return model.predict_proba(x)
        before_make,before_split=core.make_model,core.split_for_year
        core.FEATURE_GROUPS[name]=features
        core.make_model=lambda kind,features:Ready()
        core.split_for_year=lambda frame,h,y:(train,val,test)
        try:
            rr,pp,_=core.run_fold(core.Experiment(name,'hist_gradient_boosting',name),5,year,p[p.corridor.eq(KZT)])
        finally:
            core.make_model,core.split_for_year=before_make,before_split
            core.FEATURE_GROUPS.pop(name,None)
        pp=pp.merge(p[['date','corridor','rub_per_unit','session_ordinal','pr60']],on=['date','corridor'],validate='one_to_one')
        outputs.append(pp);all_cells.extend(rr)
        ckpt=HERE/'checkpoints'/f'{name}_{year}.pkl';ckpt.parent.mkdir(exist_ok=True)
        ckpt.write_bytes(pickle.dumps(model))
        fits.append({'year':year,'train_rows':len(train),'kzt_train_rows':int(train.corridor.eq(KZT).sum()),'train_min':str(train.date.min().date()),'train_max':str(train.date.max().date()),'validation_min':str(val.date.min().date()),'validation_max':str(val.date.max().date()),'test_min':str(test.date.min().date()),'test_max':str(test.date.max().date()),'validation_rows':len(val),'test_rows':len(test),'seconds':time.monotonic()-start,'checkpoint':str(ckpt.relative_to(HERE)),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'calibration':'KZT previous full year; monotone Platt or identity fallback','stage2_trees':40 if strategy.startswith('residual') else 0,'stage2_weight':getattr(model,'weight',1.),'weight_selection':'prior KZT validation Brier, fixed grid [0,.25,.5,.75,1]' if strategy=='residual_shrink' else 'fixed1'})
        print(name,year,len(train),round(time.monotonic()-start,2),flush=True)
    pred=pd.concat(outputs,ignore_index=True)
    pred.to_csv(HERE/f'{name}_predictions.csv.gz',index=False)
    pd.DataFrame(all_cells).to_csv(HERE/f'{name}_cells.csv',index=False)
    sources=[source,Path(__file__).resolve()]
    if extra_features:sources += [ROOT/'research_v4/liquidity/kase_spot_daily.csv',ROOT/'research_v4/liquidity/halyk_sell_daily.csv',ROOT/'research_v4/liquidity/experiment.py']
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(p[['date','corridor',*features]],index=False).to_numpy().tobytes()).hexdigest()
    (HERE/f'{name}_receipt.json').write_text(json.dumps({'name':name,'source_panel':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'feature_frame_fingerprint':fingerprint,'fingerprint_method':'pandas '+pd.__version__+' hash_pandas_object [date,corridor,features], index=False, uint64 bytes sha256','inputs_sha256':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},'features':features,'strategy':strategy,'stage2':'40 Newton logloss stumps depth1 leaf60 lr.025; exact float64 frozen pooled logits','fits':fits,'prediction_sha256':hashlib.sha256((HERE/f'{name}_predictions.csv.gz').read_bytes()).hexdigest()},indent=2))
    return pred

def main():
    all_predictions=[]
    with threadpool_limits(limits=1):
        for months in (24,120):
            baseline='baseline_reproduction' if months==24 else 'basis_train_120m'
            pred=pd.read_csv(ROOT/f'research_v3/models/{baseline}_h5_predictions.csv.gz',parse_dates=['date'])
            pred=pred[pred.corridor.eq(KZT)].copy();pred.config_id='v3_'+baseline
            all_predictions.append(pred)
            for strategy in ('pooled','only','residual','residual_shrink'):
                all_predictions.append(evaluate(months,strategy))
    pd.concat([old.summarize(p) for p in all_predictions],ignore_index=True).to_csv(HERE/'metrics.csv',index=False)
    pd.concat(all_predictions,ignore_index=True).to_csv(HERE/'all_predictions.csv.gz',index=False)
if __name__=='__main__':main()
