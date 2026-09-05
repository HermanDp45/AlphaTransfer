"""Export the selected historical model and an executable local research profile."""
from pathlib import Path
import sys,json,pickle,shutil,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from research_v4.final_sprint.common import fit_policy
HERE=Path(__file__).resolve().parent;DEST=ROOT/'final_solution/final_sprint'

def main():
    selection=json.loads((HERE/'selection.json').read_text());s=selection['champion'];cid=s['config_id'];cutoff='2026-01-01'
    DEST.mkdir(exist_ok=True);modeldir=DEST/'model';modeldir.mkdir(exist_ok=True)
    config=dict(config_id=cid,cutoff=cutoff,policy_name=s['policy'],selection=selection,scenario='NOW',target='R[t]<=min(R[t+1:t+5]);R=RUB_per_recipient_currency_unit',scope='KZT',artifact_role='Executable historical research profile; no contact sending',components=[])
    if cid.startswith('tabm_'):
        family=HERE/'tabm';hist=pd.read_csv(family/'policy_calibration_predictions.csv.gz',parse_dates=['date'])
        stem=cid.rsplit('_',1)[0];scope=cid.rsplit('_',1)[1]
        components=[stem.replace('sensemble3','s'+str(i)) for i in (0,1,2)] if 'sensemble3' in stem else [stem]
        features=[]
        for component in components:
            src=family/'checkpoints'/(component+'_'+cutoff);target=modeldir/component;target.mkdir(exist_ok=True)
            for f in ('weights.pt','preprocess.joblib','model.json'):shutil.copy2(src/f,target/f)
            meta=json.loads((src/'model.json').read_text());features+=meta['features'];config['components'].append(dict(path='model/'+component,weight=1/len(components),seed=meta['seed'],features=meta['features']))
        policies=pickle.loads((family/'output'/(stem+'_'+cutoff+'_policies.pkl')).read_bytes());cal=policies[scope]['calibrator']
        config['kind']='tabm'
    elif cid.startswith('hgb_'):
        src=HERE/'hgb'/(cid+'_'+cutoff+'.pkl');bundle=pickle.loads(src.read_bytes());shutil.copy2(src,modeldir/'model.pkl')
        features=bundle['features'];cal=bundle['calibrator'];config['kind']='hgb'
        hist=pd.concat([pd.read_csv(HERE/'hgb'/(cid+'_'+str(y)+'-01-01_calibration.csv.gz'),parse_dates=['date']) for y in (2025,2026)])
    elif cid.startswith('catboost_'):
        folder=HERE/'catboost'/('matched/output' if cid.endswith('_matched') else 'output');src=folder/(cid+'_'+cutoff+'.pkl');bundle=pickle.loads(src.read_bytes())
        shutil.copy2(folder/(cid+'_'+cutoff+'.cbm'),modeldir/'model.cbm');features=bundle['features'];cal=bundle['calibrator'];config['kind']='catboost'
        hist=pd.concat([pd.read_csv(HERE/'catboost'/f'candidates_{n}_predictions.csv.gz',parse_dates=['date']) for n in ('calibration','history')])
    else:raise ValueError('Selected model requires explicit export adapter: '+cid)
    hist=hist[hist.config_id.eq(cid)].copy();hist.loc[hist.split.eq('validation'),'split']='calibration'
    hist.to_csv(HERE/'selected_history.csv.gz',index=False)
    h=hist[hist.cutoff.eq(cutoff)&hist.split.eq('history')].copy();v=hist[hist.cutoff.eq(cutoff)&hist.split.eq('calibration')].copy()
    policy=fit_policy(v,h,s['policy']);config['policy']=policy;config['features']=list(dict.fromkeys(features))
    config['calibration']=dict(method=cal.method,intercept=cal.intercept,slope=cal.slope)
    config['calibration_end']=str(v.date.max().date());config['past_baseline_rate']=float(v.target.mean())
    views,_=pickle.loads((HERE/'views.pkl').read_bytes());panel=views['2010-01-01',24,1]
    pred=pd.read_csv(HERE/'selected_predictions.csv.gz',parse_dates=['date']);grid=pred[pred.cutoff.eq(cutoff)&pred['mode'].eq('normal')][['date','corridor']]
    closing=HERE/'product/results/checkpoints/closing_treasury_halyk_shrink120m_2026-01-01.pkl'
    shutil.copy2(closing,modeldir/'closing.pkl');cb=pickle.loads(closing.read_bytes());config['closing_features']=cb['features']
    # Only observed features, never targets/payoffs, cross the inference boundary.
    allfeat=list(dict.fromkeys(config['features']+cb['features']+['ret1','pr60','rub_per_unit','session_ordinal']))
    x=grid.merge(panel[['date','corridor',*allfeat]],on=['date','corridor'],validate='one_to_one')
    config['first_decision_date']=str(x.date.min().date());config['initial_processed_session']=int(x.session_ordinal.min())-1
    x.to_csv(DEST/'example_features.csv',index=False)
    for name in ('selection.json','selected_predictions.csv.gz','selected_intervals.json'):
        shutil.copy2(HERE/name,DEST/name)
    (DEST/'model.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str))
    (DEST/'requirements.txt').write_text('numpy\npandas\nscikit-learn\nscipy\ncatboost==1.2.10\ntorch\ntabm\nrtdl-num-embeddings\njoblib\nthreadpoolctl\n')
    print(json.dumps(dict(model=cid,kind=config['kind'],features=len(config['features']),output=str(DEST)),indent=2))
if __name__=='__main__':main()
