"""Execute the selected NOW model on feature rows, retaining causal contact state."""
from pathlib import Path
import os,sys,json,pickle,argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.common import select
from final_solution.training import core_experiment as core

def score(frame,config):
    features=config['features'];x=frame[features+['corridor']]
    if config['kind']=='catboost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier();model.load_model(str(HERE/'model/model.cbm'));raw=model.predict_proba(x)[:,1]
    elif config['kind']=='hgb':
        model=pickle.loads((HERE/'model/model.pkl').read_bytes())['model'];raw=model.predict_proba(x)[:,1]
    elif config['kind']=='tabm':
        import torch
        torch.set_num_threads(1)
        from research_v4.final_sprint.tabm.experiment import Neural
        raw=np.zeros(len(frame))
        for component in config['components']:
            model=Neural(component['features'],component['seed']);model.load(HERE/component['path']);raw+=component['weight']*model.predict(frame)
    else:raise ValueError(config['kind'])
    cal=config['calibration'];prob=raw if cal['method']=='identity' else expit(cal['intercept']+cal['slope']*logit(np.clip(raw,1e-6,1-1e-6)))
    return raw,prob

def run(features,output,as_of=None,state_in=None,state_out=None):
    config=json.loads((HERE/'model.json').read_text())
    # Exact decimal-to-float recovery matters before empirical quantile transforms.
    frame=pd.read_csv(features,parse_dates=['date'],float_precision='round_trip')
    required={'date','corridor','session_ordinal','ret1',*config['features'],*config['closing_features']}
    if required-set(frame):raise ValueError('Missing features: '+str(required-set(frame)))
    if frame.duplicated(['date','corridor']).any():raise ValueError('Duplicate decision rows')
    if not frame.corridor.eq('KZT').all():raise ValueError('Selected profile is validated only for KZT')
    if frame.date.min()<pd.Timestamp(config['cutoff']):raise ValueError('Input precedes model training cutoff')
    if as_of:frame=frame[frame.date.le(pd.Timestamp(as_of))].copy()
    frame=frame.sort_values(['date','corridor']).reset_index(drop=True)
    if frame.empty:raise ValueError('No eligible decision dates')
    if not np.all(np.diff(frame.session_ordinal.to_numpy())==1):raise ValueError('Feature input omits or reorders decision sessions')
    policy=config['policy'];state=policy['initial_state'];last_processed=config['initial_processed_session']
    if state_in:
        saved=json.loads(Path(state_in).read_text())
        if saved.get('config_id')!=config['config_id'] or saved.get('policy_name')!=config['policy_name'] or saved.get('model_cutoff')!=config['cutoff']:raise ValueError('State belongs to another model or policy')
        state=saved['contact_state'];last_processed=int(saved['last_processed_session'])
    # A single later date needs state from previous execution, or the full prefix.
    if not state_in and frame.date.min()!=pd.Timestamp(config['first_decision_date']):raise ValueError('Provide full historical prefix or --state-in for incremental execution')
    if frame.session_ordinal.min()!=last_processed+1:raise ValueError('Input overlaps or skips previously processed state')
    raw,prob=score(frame,config);frame['probability']=prob
    ids,next_state=select(frame,policy['threshold'],policy['cooldown'],state)
    result=frame[['date','corridor','session_ordinal']].copy();result['raw_probability']=raw;result['probability']=prob;result['candidate_signal']=result.index.isin(ids)
    closing=pickle.loads((HERE/'model/closing.pkl').read_bytes())
    cp=closing['model'].predict_proba(frame[closing['features']+['corridor']])[:,1];result['closing_probability']=core.apply_platt(closing['calibrator'],cp)
    annotation=HERE/'closing_annotation.json'
    if not annotation.exists():raise ValueError('Missing explicit CLOSING annotation policy')
    threshold=json.loads(annotation.read_text())['threshold']
    result['closing_annotation']=result.candidate_signal & result.closing_probability.ge(threshold) & frame.ret1.gt(0)
    result['primary_scenario']=np.where(result.candidate_signal,'NOW','NONE')
    result['model_verdict']=np.where(result.candidate_signal,'Модельный сигнал: выгодно сейчас','Нет сигнала')
    result.loc[result.closing_annotation,'model_verdict']+='; есть признаки закрытия окна'
    result['expires']='Recompute at next quote update; do not carry today’s signal to a later execution price.'
    result['authorized_contact']=False
    Path(output).parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    if state_out:Path(state_out).write_text(json.dumps(dict(config_id=config['config_id'],model_cutoff=config['cutoff'],policy_name=config['policy_name'],contact_state=next_state,last_processed_session=int(frame.session_ordinal.max())),indent=2))
    return dict(config_id=config['config_id'],rows=len(result),NOW_contacts=int(result.candidate_signal.sum()),CLOSING_annotations=int(result.closing_annotation.sum()),output=str(output),external_messages_sent=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',default=str(HERE/'example_features.csv'));p.add_argument('--output',default=str(HERE/'output/predictions.csv'));p.add_argument('--as-of');p.add_argument('--state-in');p.add_argument('--state-out');a=p.parse_args()
    with threadpool_limits(limits=1):print(json.dumps(run(a.features,a.output,a.as_of,a.state_in,a.state_out),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
