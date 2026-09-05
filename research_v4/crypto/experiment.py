"""Retrospective crypto ablation, frozen V3 baselines, no writes outside V4.

python research_v4/crypto/experiment.py --stage build
python research_v4/crypto/experiment.py --stage fit
python research_v4/crypto/experiment.py --stage summarize
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from research_v3.external_data.benchmark import BASIS_FEATURES,evaluate,score
from research_v3.external_data.experiment import paired_audit

OUT=HERE/'output';DATA=HERE/'data'
SOURCES={'erub':'exmo_me_USDT_RUB','ekzt':'exmo_me_USDT_KZT','ebtc':'exmo_me_BTC_USDT','brub':'binance_USDTRUB','brbtc':'binance_BTCRUB','btc':'binance_BTCUSDT','stable':'binance_USDCUSDT','bkzt':'binance_USDTKZT'}
OFFICIAL=['cbr_usd_ret1','cbr_usd_ret5','nbk_usd_ret1','nbk_usd_ret5','cbr_kzt_ret5']

def raw_features(name,filename):
    p=pd.read_csv(DATA/(filename+'.csv'),parse_dates=['source_date']).set_index('source_date').sort_index()
    original=p.copy()
    valid=(p[['open','high','low','close']].gt(0).all(axis=1)&p.volume.gt(0)&p.high.ge(p[['open','close','low']].max(axis=1))&p.low.le(p[['open','close','high']].min(axis=1)))
    quality={'source':name,'rows':len(p),'invalid_ohlcv':int((~valid).sum()),'flat_bars':int(p.high.eq(p.low).sum()),'volume_below_100_base':int(p.volume.lt(100).sum()),'extreme_range_gt50pct':int((p.high/p.low>1.5).sum())}
    # Calendar reindexing makes missing days invalidate changes/rolling windows;
    # gaps must never be treated as adjacent observations or synthetic trades.
    p.loc[~valid,['open','high','low','close','volume']]=np.nan
    p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='D'))
    log=np.log(p.close);ret=log.diff()
    f=pd.DataFrame(index=p.index)
    f['close']=p.close;f['ret1']=ret;f['ret5']=log.diff(5);f['vol20']=ret.rolling(20,min_periods=15).std()
    f['range']=np.log(p.high/p.low)
    f['volume_ratio20']=p.volume/p.volume.rolling(20,min_periods=15).median().replace(0,np.nan)
    f['amihud']=np.log1p(ret.abs()*1e6/p.volume)
    f['depeg']=log;f['max_depeg5']=log.abs().rolling(5,min_periods=3).max()
    if 'trades' in p:f['trades']=p.trades
    # Rows without actual trades are excluded from as-of joins. No observed
    # value survives more than two extra days past its availability date.
    f=f[p.close.notna()].rename(columns=lambda c:f'{name}_{c}')
    f.index.name='source_date'
    return f.reset_index(),quality

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    base=pd.read_pickle(ROOT/'research_v3/models/panel_extended.pkl')
    official=base.drop_duplicates('date')[['date','cbr_usd_level','nbk_usd_level','cbr_kzt_level']].sort_values('date')
    dates=official[['date']].copy();groups={};qualities=[];coverage=[]
    for lag in (2,5):
        daily=official.copy()
        for name,filename in SOURCES.items():
            f,q=raw_features(name,filename)
            if lag==2:qualities.append(q)
            featurecols=[c for c in f if c!='source_date']
            f['available_date']=f.source_date+pd.Timedelta(days=lag)
            # Decision-date proxy means 00:00 MSK. A UTC daily bar D closes at
            # 03:00 MSK on D+1; assigning D+2 gives >=21h extra latency.
            m=pd.merge_asof(daily[['date']],f,on=None,left_on='date',right_on='available_date',direction='backward',tolerance=pd.Timedelta(days=2))
            daily=daily.merge(m,on='date',validate='one_to_one')
            daily=daily.rename(columns={'source_date':name+'_source_date','available_date':name+'_available_date'})
            assert not (daily[name+'_available_date']>daily.date).any()
        for market,den in [('erub','cbr_usd_level'),('brub','cbr_usd_level'),('ekzt','nbk_usd_level'),('bkzt','nbk_usd_level')]:
            daily[market+'_premium']=np.log(daily[market+'_close']/daily[den])
        same_exmo=daily.erub_source_date.eq(daily.ekzt_source_date)
        same_binance=daily.brbtc_source_date.eq(daily.btc_source_date)&daily.btc_source_date.eq(daily.brub_source_date)
        same_venues=daily.erub_source_date.eq(daily.brub_source_date)
        daily['exmo_cross_premium']=np.log(daily.erub_close/daily.ekzt_close/daily.cbr_kzt_level).where(same_exmo)
        daily['binance_triangle']=np.log(daily.brbtc_close/daily.btc_close/daily.brub_close).where(same_binance)
        daily['rub_venue_dislocation']=np.log(daily.erub_close/daily.brub_close).where(same_venues)
        family={
          'global':['btc_ret1','btc_ret5','btc_vol20','btc_volume_ratio20'],
          'stable':['stable_depeg','stable_max_depeg5','stable_range'],
          'erub_premium':['erub_premium','erub_ret1','erub_ret5'],
          'erub_liquidity':['erub_range','erub_volume_ratio20','erub_amihud'],
          'ekzt_premium':['ekzt_premium','ekzt_ret1','ekzt_ret5'],
          'ekzt_liquidity':['ekzt_range','ekzt_volume_ratio20','ekzt_amihud'],
          'exmo_cross':['exmo_cross_premium'],
          'binance_rub':['brub_premium','brub_ret1','brub_ret5','brub_range','brub_volume_ratio20','binance_triangle'],
          'cross_venues':['rub_venue_dislocation'],
        }
        family['erub']=family['erub_premium']+family['erub_liquidity']
        family['exmo_joint']=family['erub']+family['ekzt_premium']+family['ekzt_liquidity']
        family['all_crypto']=family['global']+family['stable']+family['exmo_joint']
        for name,features in family.items():
            groups[name+f'_l{lag}']=[c+f'_l{lag}' for c in features]
            for year,g in daily.groupby(daily.date.dt.year):
                coverage.append({'family':name,'lag_days':lag,'year':year,'dates':len(g),'complete_row_share':g[features].notna().all(axis=1).mean(),'mean_feature_coverage':g[features].notna().mean().mean()})
        keep=[c for c in daily if c not in official]
        dates=dates.merge(daily[['date',*keep]].rename(columns={c:c+f'_l{lag}' for c in keep}),on='date',validate='one_to_one')
    dates.to_parquet(DATA/'features_daily.parquet',index=False)
    pd.DataFrame(coverage).to_csv(DATA/'feature_coverage.csv',index=False)
    pd.DataFrame(qualities).to_csv(DATA/'quality.csv',index=False)
    (OUT/'feature_groups.json').write_text(json.dumps(groups,indent=2))
    return dates,groups

def configurations(groups):
    configs=[]
    def add(name,features,years=2,scope='development_all'):
        configs.append({'model':name,'features':features,'train_years':years,'scope':scope})
    add('basis_short',[]);add('basis_long',[],10)
    for f in ('global','stable','erub_premium','erub_liquidity','erub','ekzt_premium','ekzt_liquidity','exmo_cross','exmo_joint','all_crypto'):
        add('short_'+f,groups[f+'_l2'])
    for f in ('global','erub','exmo_joint','all_crypto'):
        add('long_'+f,groups[f+'_l2'],10)
    for years in (2,10):
        title='short' if years==2 else 'long'
        add(title+'_official_controls',OFFICIAL,years)
        add(title+'_official_plus_crypto',OFFICIAL+groups['exmo_joint_l2'],years)
        add(title+'_exmo_joint_lag5',groups['exmo_joint_l5'],years)
    add('short_binance_rub',groups['binance_rub_l2'],scope='matched_pre_delisting')
    add('short_cross_venues',groups['cross_venues_l2'],scope='matched_pre_delisting')
    # Bounded follow-up after the first matrix: isolate dimensionality and
    # timing. These remain explicitly post-selection, not a held-out test.
    add('long_erub_premium',groups['erub_premium_l2'],10,scope='posthoc_compact')
    add('long_erub_premium_only',['erub_premium_l2'],10,scope='posthoc_compact')
    add('long_ekzt_premium',groups['ekzt_premium_l2'],10,scope='posthoc_compact')
    add('long_exmo_cross',groups['exmo_cross_l2'],10,scope='posthoc_compact')
    add('short_erub_premium_lag5',groups['erub_premium_l5'],scope='posthoc_timing')
    add('long_erub_premium_lag5',groups['erub_premium_l5'],10,scope='posthoc_timing')
    return configs

def feature_panel(extended=False):
    path=ROOT/'research_v3/models'/('panel_extended.pkl' if extended else 'panel_v2.pkl')
    # Explicit frozen cache read avoids rebuilding or mutating sealed V3.
    return pd.read_pickle(path).merge(pd.read_parquet(DATA/'features_daily.parquet'),on='date',how='left',validate='many_to_one')

def fit():
    groups=json.loads((OUT/'feature_groups.json').read_text());configs=configurations(groups)
    (OUT/'configurations.json').write_text(json.dumps(configs,indent=2))
    panels={False:feature_panel(),True:feature_panel(True)}
    for c in configs:
        cid=c['model'];path=OUT/(cid+'_predictions.csv.gz');cellpath=OUT/(cid+'_cells.csv')
        if path.exists() and cellpath.exists():continue
        print('FIT',cid,flush=True);beg=time.time()
        pred,cells=evaluate(panels[c['train_years']==10],BASIS_FEATURES+c['features'],cid,train_years=c['train_years'],years=(2023,2024,2025,2026),disable_early_stopping=c['train_years']==10)
        pred.to_csv(path,index=False);cells.to_csv(cellpath,index=False)
        print('DONE',cid,round(time.time()-beg,2),flush=True)

def basic_metric(g):
    y=g.target.to_numpy();p=g.probability.to_numpy();sel=g.candidate_signal.astype(bool)
    cell=g.groupby(['fold_test_year','corridor']).target.transform('mean')
    expected=float(cell[sel].sum());hits=float(g.loc[sel,'target'].sum())
    # Standardize the reference utility against each selected signal's year and
    # corridor. This is a reference-rate hindsight metric, not actual savings.
    avg=g.groupby(['fold_test_year','corridor']).forward_bps.transform('mean')
    return {'rows':len(g),'dates':g.date.nunique(),'brier':float(np.mean((p-y)**2)),'candidate_count':int(sel.sum()),'candidate_hit_rate':hits/sel.sum() if sel.sum() else np.nan,'candidate_lift':hits/expected if expected else np.nan,'candidate_forward_delta_bps':float((g.forward_bps-avg)[sel].mean())}

def summarize():
    configs=json.loads((OUT/'configurations.json').read_text());preds=[];cells=[];verification=[]
    for c in configs:
        path=OUT/(c['model']+'_predictions.csv.gz')
        if not path.exists():continue
        p=pd.read_csv(path,parse_dates=['date']);preds.append(p)
        cells.append(pd.read_csv(OUT/(c['model']+'_cells.csv')))
        if c['model'] in ('basis_short','basis_long'):
            original='baseline_reproduction' if c['model']=='basis_short' else 'basis_train_120m'
            old=pd.read_csv(ROOT/'research_v3/models'/(original+'_h5_predictions.csv.gz'),parse_dates=['date'])
            m=p.merge(old,on=['date','corridor','fold_test_year'],suffixes=('_new','_old'),validate='one_to_one')
            delta=(m.probability_new-m.probability_old).abs().max()
            assert len(m)==len(p)==len(old) and delta<1e-12,(original,delta,len(m))
            assert (m.target_new==m.target_old).all()
            assert (m.candidate_signal_new==m.candidate_signal_old).all()
            verification.append({'model':c['model'],'baseline':original,'rows':len(m),'max_probability_diff':delta,'candidate_signals_exact':True})
    p=pd.concat(preds,ignore_index=True);cells=pd.concat(cells,ignore_index=True)
    bymodel={c['model']:c for c in configs};ledger=[];cis=[]
    for scope,years in [('development',[2023,2024,2025]),('diagnostic_2026',[2026]),('matched_binance_2023',[2023])]:
        selected=p[p.fold_test_year.isin(years)]
        score(selected,cells[cells.fold_test_year.isin(years)]).to_csv(OUT/(scope+'_metrics_full.csv'),index=False)
        pairs=[]
        for cid,g in selected.groupby('config_id'):
            c=bymodel[cid];row=basic_metric(g);row.update(model=cid,scope=scope,train_years=c['train_years'],feature_count=len(c['features']))
            old=selected[selected.config_id.eq('basis_short')];long=selected[selected.config_id.eq('basis_long')]
            row['brier_delta_vs_original']=row['brier']-basic_metric(old)['brier'];row['relative_brier_gain_vs_original']=-row['brier_delta_vs_original']/basic_metric(old)['brier']
            row['brier_delta_vs_long']=row['brier']-basic_metric(long)['brier']
            row['flags']='retrospective_post_selection;not_production;spot_not_P2P;synthetic_excluded'
            if 'binance' in cid or 'cross_venues' in cid:row['flags']+=';RUB_delisted_after_2024_01_30;2023_is_only_full_matched_test_year'
            ledger.append(row)
            if cid!='basis_short':pairs.append((cid,'basis_short'))
            if c['train_years']==10 and cid!='basis_long':pairs.append((cid,'basis_long'))
        for prefix in ('short','long'):
            pairs.append((prefix+'_official_plus_crypto',prefix+'_official_controls'))
        ci=paired_audit(selected,pairs,reps=10000);ci['scope']=scope;cis.append(ci)
    pd.DataFrame(ledger).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(cis,ignore_index=True).to_csv(OUT/'paired_ci.csv',index=False)
    # Also compare identical decision rows where every candidate add-on feature
    # is observed. The fitted models and thresholds are unchanged. Every row
    # carries its own matched baseline; these scores are not interchangeable
    # with the full 2023-25 benchmark, especially after a market disappears.
    joined=feature_panel(True).set_index(['date','corridor'])
    matched_rows=[];matched_cis=[]
    for c in configs:
        if not c['features']:continue
        mask=joined[c['features']].notna().all(axis=1)
        keys=mask[mask].reset_index()[['date','corridor']]
        for period,years in [('development',[2023,2024,2025]),('diagnostic_2026',[2026])]:
            a=p[p.config_id.eq(c['model'])&p.fold_test_year.isin(years)]
            a=a.merge(keys,on=['date','corridor'],validate='one_to_one')
            if a.empty:continue
            for baseline in ['basis_short','basis_long']:
                b=p[p.config_id.eq(baseline)].merge(a[['date','corridor','fold_test_year']],on=['date','corridor','fold_test_year'],validate='one_to_one')
                met=basic_metric(a);bm=basic_metric(b)
                matched_rows.append({'model':c['model'],'baseline':baseline,'period':period,**met,'baseline_brier_matched':bm['brier'],'brier_delta_matched':met['brier']-bm['brier'],'baseline_candidate_lift_matched':bm['candidate_lift'],'support_rule':'all added features observed; no refit or retuning; decision rows identical'})
                ci=paired_audit(pd.concat([a,b]),[(c['model'],baseline)],reps=10000)
                ci['scope']=period+'_source_available';matched_cis.append(ci)
    pd.DataFrame(matched_rows).to_csv(OUT/'source_available_matched_metrics.csv',index=False)
    pd.concat(matched_cis,ignore_index=True).to_csv(OUT/'source_available_matched_ci.csv',index=False)
    yearly=[]
    for (cid,year,corridor),g in p.groupby(['config_id','fold_test_year','corridor']):
        yearly.append({'model':cid,'year':year,'corridor':corridor,**basic_metric(g)})
    pd.DataFrame(yearly).to_csv(OUT/'year_corridor_metrics.csv',index=False)
    (OUT/'baseline_verification.json').write_text(json.dumps(verification,indent=2))
    manifest={'protocol':'h5 CBR-row effective-date proxy, unchanged annual core folds/calibration/candidate policy','train_years':[2,10],'validation_years':1,'development_years':[2023,2024,2025],'diagnostic_year':2026,'crypto_UTC_bar_open_to_availability_lag_days':[2,5],'asof_tolerance_after_availability_days':2,'imputed_cryptomarket_prices':False,'source_snapshot_retrieved_in_2026_not_historical_vintages':True,'long_disable_early_stopping':True,'number_of_configurations':len(configs),'expected_annual_fits':4*len(configs),'no_model_promotion':True,'sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'research_v3/external_data/benchmark.py',ROOT/'final_solution/training/core_experiment.py',ROOT/'final_solution/training/train_and_evaluate.py',*sorted(DATA.glob('*.csv')),*sorted(DATA.glob('*.parquet'))]}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['build','fit','summarize','all'],default='all');a=ap.parse_args()
    if a.stage in ('build','all'):build()
    if a.stage in ('fit','all'):fit()
    if a.stage in ('summarize','all'):summarize()

if __name__=='__main__':main()
