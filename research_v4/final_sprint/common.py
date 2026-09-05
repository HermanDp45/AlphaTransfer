"""Causal notification policies and weighted shallow residual model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.tree import DecisionTreeRegressor
from final_solution.training import core_experiment as core

class WeightedAdapted:
    def __init__(self, pooled, adapt=True):
        self.pooled=pooled; self.adapt=adapt; self.weight=1.
    def fit(self,x,y,sample_weight=None):
        w=np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight)
        self.pooled.fit(x,y,classifier__sample_weight=w)
        self.trees=[]
        if not self.adapt:return self
        mask=x.corridor.eq('KZT').to_numpy()
        z=self.pooled.named_steps['preprocessor'].transform(x.loc[mask])
        labels=np.asarray(y)[mask]; w=w[mask]
        margin=logit(self.pooled.predict_proba(x.loc[mask])[:,1].clip(1e-7,1-1e-7))
        for step in range(40):
            p=expit(margin); h=(p*(1-p)).clip(1e-5)
            tree=DecisionTreeRegressor(max_depth=1,min_samples_leaf=60,random_state=core.SEED+step)
            tree.fit(z,(labels-p)/h,sample_weight=h*w)
            margin+=.025*tree.predict(z);self.trees.append(tree)
        return self
    def predict_proba(self,x):
        orig=self.pooled.predict_proba(x)
        if not self.trees:return orig
        z=self.pooled.named_steps['preprocessor'].transform(x)
        correction=sum(.025*t.predict(z) for t in self.trees)
        q=expit(logit(orig[:,1].clip(1e-7,1-1e-7))+correction)
        p=(1-self.weight)*orig[:,1]+self.weight*q
        return np.column_stack([1-p,p])

def select(frame,threshold,cooldown=3,initial=None):
    """At most two contacts/calendar week; threshold + session cooldown, no quota fill."""
    state={} if initial is None else {k:dict(v) for k,v in initial.items()}
    selected=[]
    for row in frame.sort_values(['date','corridor']).itertuples():
        c=str(row.corridor);week=str(row.date.to_period('W-SUN'))
        st=state.setdefault(c,dict(last=-10000,week=week,count=0))
        if st['week']!=week:st.update(week=week,count=0)
        t=threshold[c] if isinstance(threshold,dict) else threshold
        if np.isfinite(row.probability) and row.probability>=t and row.session_ordinal-st['last']>cooldown and st['count']<2:
            selected.append(row.Index);st.update(last=int(row.session_ordinal),count=st['count']+1)
    return np.array(selected,dtype=int),state

def metrics(g):
    selected=g[g.candidate_signal.astype(bool)]
    counts=g.groupby(g.date.dt.to_period('W-SUN')).candidate_signal.sum()
    counts=counts.reindex(pd.period_range(counts.index.min(),counts.index.max(),freq='W-SUN'),fill_value=0)
    base=float(g.target.mean())
    return dict(rows=len(g),dates=g.date.nunique(),signals=len(selected),weeks=len(counts),
        weeks_1_2=float(counts.between(1,2).mean()),signals_per_week=float(counts.mean()),silent_weeks=int(counts.eq(0).sum()),
        brier=float(np.mean((g.probability-g.target)**2)),base_hit=base,hit_rate=float(selected.target.mean()),
        lift=float(selected.target.mean()/base) if base else float('nan'),
        forward_delta_bps=float(selected.forward_bps.mean()-g.forward_bps.mean()),
        forward_signal_bps=float(selected.forward_bps.mean()))

POLICIES={'legacy':None,'cadence90_cd3':(.9,3),'cadence90_cd2':(.9,2),'cadence85_cd2':(.85,2)}

def fit_policy(validation,history,name):
    if name=='legacy':
        thresholds,_,_=core.choose_frequency_threshold(validation,validation.probability.to_numpy())
        cooldown=3
    else:
        coverage,cooldown=POLICIES[name];thresholds={}
        for c,g in validation.groupby('corridor'):
            candidates=[]
            # Finite declared quantile grid. Outcome-informed threshold uses only matured calibration labels.
            for q in np.arange(0,.81,.05):
                t=float(g.probability.quantile(q));ids,_=select(g,t,cooldown)
                z=g.copy();z['candidate_signal']=z.index.isin(ids);m=metrics(z)
                candidates.append((m['weeks_1_2']>=coverage,m['lift'],m['forward_delta_bps'],m['weeks_1_2'],t))
            feasible=[x for x in candidates if x[0]]
            best=max(feasible,key=lambda x:(x[1],x[2],x[3])) if feasible else max(candidates,key=lambda x:(x[3],x[1]))
            thresholds[str(c)]=best[-1]
    _,state=select(history,thresholds,cooldown)
    return dict(name=name,threshold=thresholds,cooldown=cooldown,initial_state=state)

def apply_policy(test,policy):
    q=test.copy();ids,_=select(q,policy['threshold'],policy['cooldown'],policy['initial_state'])
    q['candidate_signal']=q.index.isin(ids);q['policy']=policy['name'];return q
