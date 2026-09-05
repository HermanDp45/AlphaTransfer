"""Optional independent CLOSING H3 diagnostic; never substitutes NOW probability."""
from pathlib import Path
import joblib,numpy as np
from scipy.special import expit
from .features import FEATURES
from .model import sha


def score_annotation(frame,now_candidates,config,base):
    c=config.get('closing')
    if not c:return {'status':'unavailable','enabled':False,'reason':'No separately validated CLOSING head configured','train_horizon':None},None,None,None
    if c.get('scenario')!='CLOSING' or c.get('train_horizon')!=3 or c.get('target')!='R[t+3]>R[t]' or c.get('features')!=FEATURES:raise ValueError('CLOSING must have its own exact H3 endpoint contract')
    if c.get('requires_now') is not True or c.get('requires_positive_ret1') is not True or c.get('extra_contacts') is not False:raise ValueError('CLOSING cannot add contacts or replace the NOW gate')
    if c.get('enabled') and c.get('status','').startswith('diagnostic_only'):raise ValueError('Diagnostic-only failed policy cannot be enabled implicitly')
    path=Path(base)/c['model']
    if sha(path)!=c['model_sha256']:raise ValueError('CLOSING artifact checksum mismatch')
    model=joblib.load(path);raw=model.predict_proba(frame[FEATURES+['corridor']])[:,1].astype(np.float64)
    cal=c['calibration']
    if cal['method']=='identity':prob=raw
    else:
        if cal['method']!='prior_year_monotone_platt' or not np.isfinite([cal['intercept'],cal['slope']]).all() or cal['slope']<=0:raise ValueError('Invalid independent CLOSING calibration')
        clipped=np.clip(raw,1e-6,1-1e-6);prob=expit(cal['intercept']+cal['slope']*np.log(clipped/(1-clipped)))
    condition=np.asarray(now_candidates,bool)&(prob>=float(c['threshold']))&frame.ret1.gt(0).to_numpy()
    annotation=condition&bool(c.get('enabled',False))
    metadata={'status':c.get('status','configured'),'enabled':bool(c.get('enabled',False)),'scenario':'CLOSING','train_horizon':3,'target_contract':'CLOSING:R[t+3]>R[t];R=RUB_per_KZT;h=3_effective_CBR_rows;tau=0','model_sha256':c['model_sha256'],'threshold':float(c['threshold']),'extra_contacts':0,'own_probability':True,'annual_metrics':c.get('annual_metrics',[])}
    return metadata,prob,condition,annotation
