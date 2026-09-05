#!/usr/bin/env python3
"""Read foundation results, add honest historical-path baselines and paired CIs."""
from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, log_loss
from benchmark import OUT, ROOT, H, CONTEXT, SEED, core, load_panel, load_history, forecast_metrics, save_json, sha

sys.path.insert(0,str(ROOT/'research_v3/models'))
from decision_uncertainty import aggregate_block_arrays, block_weights, evaluate_sums, cadence


def baselines():
    p,(wide,logs)=load_panel(),load_history()
    dates=pd.DatetimeIndex(sorted(p.date.unique()))
    grid=np.array([.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
    qs,qg,ps=[],[],[]
    for date in dates:
        i=wide.index.get_loc(date)
        context=logs[:,i-CONTEXT+1:i+1]
        windows=np.lib.stride_tricks.sliding_window_view(context,H+1,axis=1)
        paths=windows[:,:,1:]-windows[:,:,:1]
        offsets=np.quantile(paths,grid,axis=1).transpose(1,0,2)
        qs.append(context[:,-1,None,None]+offsets)
        sigma=np.std(np.diff(context[:,-61:],axis=1),axis=1,ddof=1)
        qg.append(context[:,-1,None,None]+sigma[:,None,None]*norm.ppf(grid)[None,:,None]*np.sqrt(np.arange(1,H+1))[None,None,:])
        ps.append((paths.min(axis=-1)>=-1e-12).mean(axis=1))
    keys=pd.MultiIndex.from_product([dates,wide.columns],names=['date','corridor']).to_frame(index=False)
    keys['path_p']=np.asarray(ps).ravel()
    p=p.merge(keys,on=['date','corridor'],validate='one_to_one')
    class HistoricalProbability:
        def fit(self,x,y): return self
        def predict_proba(self,x):
            z=x.path_p.to_numpy()
            return np.column_stack((1-z,z))
    old=core.make_model
    core.make_model=lambda kind,fs:HistoricalProbability()
    core.FEATURE_GROUPS['historical_path']=['path_p']
    try:
        for year in [2023,2024,2025,2026]:
            _,_,test=core.split_for_year(p,H,year)
            rows,oot,_=core.run_fold(core.Experiment('historical_path','hist_gradient_boosting','historical_path'),H,year,p)
            oot.to_csv(OUT/'output'/f'historical_path_{year}_predictions.csv.gz',index=False)
            pd.DataFrame(rows).to_csv(OUT/'output'/f'historical_path_{year}_cells.csv',index=False)
            for name,q in [('historical_path',qs),('random_walk_gaussian60',qg)]:
                forecast_metrics(wide,logs,dates,np.asarray(q),grid,year,name,test[['date','corridor']]).to_csv(OUT/'output'/f'{name}_{year}_forecast_losses.csv',index=False)
    finally:
        core.make_model=old


def score(frame):
    selected=frame.candidate_signal.astype(bool)
    sums,signals,months=aggregate_block_arrays(frame,{'x':selected})
    result={k:float(v) for k,v in evaluate_sums(sums.sum(axis=0),signals['x'].sum(axis=0)).items()}
    y,p=frame.target.to_numpy(),frame.probability.to_numpy()
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.sum(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i])) for i in np.unique(bins))/len(frame)
    result.update(brier=np.mean((y-p)**2),logloss=log_loss(y,p),auc=roc_auc_score(y,p),ece10=ece,
                  rows=len(frame),dates=frame.date.nunique(),**cadence(frame,selected))
    return result


def paired(frame, other, repetitions=10000):
    keys=['date','corridor']
    assert frame[keys].equals(other[keys])
    assert np.allclose(frame.target,other.target)
    # Common month resamples keep all corridors and recompute cell baselines.
    masks={'candidate':frame.candidate_signal.to_numpy(bool),'baseline':other.candidate_signal.to_numpy(bool)}
    exposure,signals,months=aggregate_block_arrays(frame,masks)
    weights=block_weights(months,repetitions)
    n=exposure.shape[1]
    ee=(weights@exposure.reshape(len(months),-1)).reshape(repetitions,n,5)
    outcomes={k:evaluate_sums(ee,(weights@v.reshape(len(months),-1)).reshape(repetitions,n,5)) for k,v in signals.items()}
    delta=(frame.probability-frame.target)**2-(other.probability-other.target)**2
    monthly=pd.DataFrame({'date':frame.date,'delta':delta}).assign(month=frame.date.dt.to_period('M')).groupby('month').delta.agg(['sum','count'])
    draws=(weights@monthly.loc[months,'sum'].to_numpy())/(weights@monthly.loc[months,'count'].to_numpy())
    row={'delta_brier':float(delta.mean()),'delta_brier_ci95_low':np.quantile(draws,.025),'delta_brier_ci95_high':np.quantile(draws,.975),'repetitions':repetitions}
    point_candidate,point_baseline=score(frame),score(other)
    for metric in ('lift','forward_delta_bps','symmetric_delta_bps','regret_bps','signals'):
        diff=outcomes['candidate'][metric]-outcomes['baseline'][metric]
        lo,hi=np.nanquantile(diff,[.025,.975])
        point=point_candidate[metric]-point_baseline[metric]
        row.update({metric+'_delta':point,metric+'_delta_ci95_low':lo,metric+'_delta_ci95_high':hi})
    return row


def main():
    baselines()
    frames=[]
    for path in sorted((OUT/'output').glob('*_predictions.csv.gz')):
        p=pd.read_csv(path,parse_dates=['date'])
        p['config_id']=path.name.rsplit('_',2)[0]
        frames.append(p)
    for name in ['baseline_reproduction','basis_train_120m']:
        p=pd.read_csv(ROOT/'research_v3/models'/f'{name}_h5_predictions.csv.gz',parse_dates=['date'])
        p['config_id']=name
        frames.append(p)
    allp=pd.concat(frames,ignore_index=True)
    allp.to_csv(OUT/'predictions.csv.gz',index=False)
    cells=[]
    for (name,year,corridor),cell in allp.groupby(['config_id','fold_test_year','corridor']):
        cells.append({'config_id':name,'fold_test_year':year,'corridor':corridor,**score(cell)})
    pd.DataFrame(cells).to_csv(OUT/'cells.csv',index=False)
    rows,intervals=[] ,[]
    for stage, years in [('development_2023_2025',[2023,2024,2025]),('inspected_diagnostic_2026',[2026])]:
        data=allp[allp.fold_test_year.isin(years)]
        for scope in ['all','KZT']:
            scoped=data if scope=='all' else data[data.corridor.eq('KZT')]
            grouped={k:v.sort_values(['date','corridor']).reset_index(drop=True) for k,v in scoped.groupby('config_id')}
            for name,p in grouped.items():
                if '_kzt' in name and scope!='KZT':
                    continue
                rows.append({'stage':stage,'scope':scope,'config_id':name,**score(p)})
                candidates=['baseline_reproduction','basis_train_120m']
                if '_kzt' in name:
                    candidates.append(name.replace('_kzt',''))
                if '_ft' in name and '_kzt' not in name:
                    candidates.append(name.replace('_ft','_zs'))
                for baseline in dict.fromkeys(candidates):
                    if name==baseline or baseline not in grouped:
                        continue
                    intervals.append({'stage':stage,'scope':scope,'config_id':name,'benchmark':baseline,**paired(p,grouped[baseline])})
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    losses=pd.concat([pd.read_csv(f,parse_dates=['date']) for f in sorted((OUT/'output').glob('*_forecast_losses.csv'))],ignore_index=True)
    losses.to_csv(OUT/'forecast_losses.csv.gz',index=False)
    loss_rows=[]
    for (name,year,corridor),f in losses.groupby(['config_id','fold_test_year','corridor']):
        loss_rows.append({'config_id':name,'fold_test_year':year,'corridor':corridor,'rows':len(f),**f[['pinball_bps','mae_bps','coverage80']].mean().to_dict()})
    pd.DataFrame(loss_rows).to_csv(OUT/'forecast_cells.csv',index=False)
    forecast_summary,forecast_intervals=[],[]
    for stage,years in [('development_2023_2025',[2023,2024,2025]),('inspected_diagnostic_2026',[2026])]:
        data=losses[losses.fold_test_year.isin(years)]
        for scope in ['all','KZT']:
            scoped=data if scope=='all' else data[data.corridor.eq('KZT')]
            grouped={k:v.sort_values(['date','corridor']).reset_index(drop=True) for k,v in scoped.groupby('config_id')}
            for name,f in grouped.items():
                if '_kzt' in name and scope!='KZT':
                    continue
                forecast_summary.append({'stage':stage,'scope':scope,'config_id':name,'rows':len(f),**f[['pinball_bps','mae_bps','coverage80']].mean().to_dict()})
                alternatives=['random_walk_gaussian60']
                if '_ft' in name and '_kzt' not in name:
                    alternatives.append(name.replace('_ft','_zs'))
                if '_kzt' in name:
                    alternatives.append(name.replace('_kzt',''))
                for baseline in alternatives:
                    if baseline==name or baseline not in grouped:
                        continue
                    other=grouped[baseline]
                    assert f[['date','corridor']].equals(other[['date','corridor']])
                    row={'stage':stage,'scope':scope,'config_id':name,'benchmark':baseline,'repetitions':10000}
                    for metric in ['pinball_bps','mae_bps','coverage80']:
                        delta=f[metric]-other[metric]
                        blocks=pd.DataFrame({'month':f.date.dt.to_period('M'),'delta':delta}).groupby('month').delta.agg(['sum','count'])
                        weights=block_weights(list(blocks.index),10000)
                        draws=(weights@blocks['sum'].to_numpy())/(weights@blocks['count'].to_numpy())
                        row.update({metric+'_delta':delta.mean(),metric+'_delta_ci95_low':np.quantile(draws,.025),metric+'_delta_ci95_high':np.quantile(draws,.975)})
                    forecast_intervals.append(row)
    pd.DataFrame(forecast_summary).to_csv(OUT/'forecast_summary.csv',index=False)
    pd.DataFrame(forecast_intervals).to_csv(OUT/'forecast_paired_intervals.csv',index=False)
    save_json(OUT/'assessment_receipt.json',{'bootstrap':'year-stratified whole-month paired blocks, all currencies retained; cell baselines recomputed per draw','repetitions':10000,'selection':'none on test; all reported trials exploratory','input_files':{str(f.relative_to(OUT)):sha(f) for f in sorted((OUT/'output').glob('*_predictions.csv.gz'))},'code_sha256':sha(__file__)})
    print(pd.DataFrame(rows).query("stage=='development_2023_2025' and scope=='all'")[['config_id','brier','lift','forward_delta_bps','signals']].to_string(index=False))


if __name__=='__main__':
    main()
