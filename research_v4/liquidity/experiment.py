"""Open market/bank quote ablations on the frozen annual V3 sample.

No OOT hyperparameter search; predeclared source families, HGB settings, lags.
Daily KASE turnover is real volume, not order imbalance or bank execution.
"""
from __future__ import annotations
from pathlib import Path
import os,sys,json,hashlib,pickle,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v3.external_data import benchmark
from research_v3.models import experiment as old
from final_solution.training import core_experiment as core
HERE=Path(__file__).resolve().parent

def join(p,table,prefix,lag):
    table=table.copy().sort_values('observation_date')
    table['available_date']=table.observation_date+pd.Timedelta(days=lag)
    table=table.rename(columns={c:prefix+c for c in table if c!='available_date'})
    result=pd.merge_asof(p.sort_values('date'),table.sort_values('available_date'),left_on='date',right_on='available_date',direction='backward',tolerance=pd.Timedelta(days=7))
    result=result.rename(columns={'available_date':prefix+'available_date'})
    observed=result[prefix+'observation_date'].notna()
    assert (result.loc[observed,prefix+'available_date']<=result.loc[observed,'date']).all()
    assert ((result.loc[observed,'date']-result.loc[observed,prefix+'observation_date']).dt.days>=lag).all()
    return result

def build_panel(extended=False,lag=1):
    p=pd.read_pickle(ROOT/'research_v3/models'/('panel_extended.pkl' if extended else 'panel_v2.pkl'))
    raw=pd.read_csv(HERE/'kase_spot_daily.csv')
    raw['observation_date']=pd.to_datetime(raw.date_trade).dt.normalize()
    # Repeated sessions are cumulative daily results; never sum them as volume.
    dup=raw[raw.duplicated(['observation_date','code'],keep=False)]
    duplicate_conflicts=int((dup.groupby(['observation_date','code'])[['volume','average','deals']].nunique()>1).any(axis=1).sum())
    raw['completeness']=(raw.average.gt(0)&raw.volume.gt(0)&raw.deals.gt(0)).astype(int)*2+(raw.high.gt(0)&raw.low.gt(0)).astype(int)
    # Later session placeholders can omit trades/range. Prefer a complete daily
    # record, then latest session; never add cumulative session observations.
    raw=raw.sort_values(['observation_date','code','completeness','num_sess'],na_position='first').drop_duplicates(['observation_date','code'],keep='last')
    prices=[];liquidity=[];coverage=[]
    for symbol in ('RUBKZT_TOM','USDKZT_TOM','CNYKZT_TOM'):
        f=raw[raw.code.eq(symbol)].sort_values('observation_date').copy()
        prefix='kase_'+symbol.lower()+'_'
        active=f.average.gt(0)&f.volume.gt(0)&f.deals.gt(0)
        # Returns use active observations; no return is fabricated on no-trade days.
        t=f[active].copy();log=np.log(t.average)
        q=pd.DataFrame({'observation_date':t.observation_date,'log_price':log,'ret1':log.diff(),'ret5':log.diff(5),'close_vs_vwap':np.log(t['last'].where(t['last']>0)/t.average),'range':np.log(t.high.where(t.high>0)/t.low.where(t.low>0)),'log_volume':np.log1p(t.volume),'volume_ratio20':t.volume/t.volume.rolling(20,min_periods=10).median(),'trades_ratio20':t.deals/t.deals.rolling(20,min_periods=10).median(),'log_illiquidity20':np.log1p((log.diff().abs()/t.volume).rolling(20,min_periods=10).mean()*1e9)})
        p=join(p,q,prefix,lag)
        if symbol!='CNYKZT_TOM':
            prices.extend(prefix+c for c in ('ret1','ret5','close_vs_vwap'))
            liquidity.extend(prefix+c for c in ('range','log_volume','volume_ratio20','trades_ratio20','log_illiquidity20'))
        coverage.append({'source':symbol,'raw_days':len(f),'active_days':int(active.sum()),'min':str(f.observation_date.min().date()),'max':str(f.observation_date.max().date()),'nonpositive_omitted':int((~active).sum()),'crossed_daily_bid_offer_share':float((f.bid>f.offer).mean())})
    # Known current official KZT converted to same KZT per RUB orientation.
    kzt=p[p.corridor.eq('KZT')][['date','rub_per_unit']].rename(columns={'rub_per_unit':'known_kzt_rub_per_unit'})
    p=p.merge(kzt,on='date',how='left',validate='many_to_one')
    p['kase_rub_market_official_basis']=p['kase_rubkzt_tom_log_price']+np.log(p.known_kzt_rub_per_unit)
    prices+=['kase_rub_market_official_basis']
    bank=pd.read_csv(HERE/'halyk_sell_daily.csv',parse_dates=['date'])
    for (client,ccy),f in bank.groupby(['client','currency']):
        f=f.sort_values('date');log=np.log(f.value.where(f.value>0))
        p=join(p,pd.DataFrame({'observation_date':f.date,'log_price':log,'ret1':log.diff(),'ret5':log.diff(5)}),f'halyk_{client}_{ccy.lower()}_',lag)
    p['halyk_rub_sell_official_basis']=p.halyk_personal_rub_log_price+np.log(p.known_kzt_rub_per_unit)
    same=p.halyk_personal_rub_observation_date.eq(p.halyk_legal_rub_observation_date)
    p['halyk_rub_personal_legal_gap']=(p.halyk_personal_rub_log_price-p.halyk_legal_rub_log_price).where(same)
    same=p.halyk_personal_rub_observation_date.eq(p.kase_rubkzt_tom_observation_date)
    p['halyk_rub_sell_market_gap']=(p.halyk_personal_rub_log_price-p.kase_rubkzt_tom_log_price).where(same)
    halyk=['halyk_rub_sell_official_basis','halyk_rub_personal_legal_gap','halyk_personal_rub_ret1','halyk_personal_rub_ret5','halyk_personal_usd_ret1','halyk_personal_usd_ret5']
    moex=['moex_cnyrub_range','moex_cnyrub_trades_ratio20','moex_cnyrub_log_trades','moex_rusfar_cny_minus_rub']
    # Existing MOEX columns already have their own V3 lag1 receipt. Apply one
    # additional effective-row lag for conservative sensitivity, not relabeling.
    if lag==2:
        for col in moex:p[col]=p.sort_values(['corridor','date']).groupby('corridor')[col].shift(1).reindex(p.index)
    groups={'kase_prices':prices,'kase_liquidity':liquidity,'kase_full':prices+liquidity,'halyk':halyk,'moex_liquidity':moex,'kase_halyk':prices+liquidity+halyk+['halyk_rub_sell_market_gap']}
    p=p.sort_values(['date','corridor']).reset_index(drop=True)
    report={'lag_calendar_days':lag,'KASE_duplicate_conflict_cells':duplicate_conflicts,'KASE_duplicate_resolution':'complete active price+volume+deals first, valid range second, latest numbered session as tie-break; never sum','sources':coverage,'bank_price_direction':'BANK SELLS RUB; not customer RUB->KZT all-in quote','feature_groups':groups,'development_nonmissing':{k:float(p[p.date.between('2023-01-01','2025-12-31')][k].notna().mean()) for k in set(sum(groups.values(),[]))},'inputs_sha256':{str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in [HERE/'kase_spot_daily.csv',HERE/'halyk_sell_daily.csv',Path(__file__).resolve(),ROOT/'research_v3/models'/('panel_extended.pkl' if extended else 'panel_v2.pkl')]}}
    report['source_timing']={'KASE':'trade date + '+str(lag)+' calendar days','Halyk':'chart effective date + '+str(lag)+' calendar days','MOEX_additions':'V3 available_date lag1' if lag==1 else 'V3 lag1 plus one effective CBR row, not exactly two calendar days','incumbent_CNY_basis':'unchanged V3 same-session basis with original lag1; not stressed by new-source lag test'}
    return p,groups,report

def run():
    outputs=[];summaries=[];receipts=[]
    with threadpool_limits(limits=1):
        for years in (2,10):
            for lag in (1,2):
                p,groups,report=build_panel(years==10,lag)
                (HERE/f'feature_receipt_{years}y_lag{lag}.json').write_text(json.dumps(report,indent=2))
                p.to_pickle(HERE/f'panel_{years}y_lag{lag}.pkl')
                for group,features in groups.items():
                    name=f'{group}_{years*12}m_lag{lag}';models=[]
                    old_factory=core.make_model
                    def factory(kind,fs):
                        model=old_factory(kind,fs);models.append(model);return model
                    core.make_model=factory
                    start=time.monotonic()
                    try:pred,cells=benchmark.evaluate(p,old.BASE_FEATURES+[old.BASIS]+features,name,train_years=years,disable_early_stopping=True)
                    finally:core.make_model=old_factory
                    pred=pred.merge(p[['date','corridor','rub_per_unit','session_ordinal','pr60']],on=['date','corridor'],validate='one_to_one')
                    dest=HERE/'predictions'/f'{name}.csv.gz';dest.parent.mkdir(exist_ok=True);pred.to_csv(dest,index=False)
                    ckpt=HERE/'checkpoints'/f'{name}.pkl';ckpt.parent.mkdir(exist_ok=True);ckpt.write_bytes(pickle.dumps(models))
                    cells.to_csv(HERE/'predictions'/f'{name}_cells.csv',index=False)
                    receipts.append({'name':name,'features':old.BASE_FEATURES+[old.BASIS]+features,'train_years':years,'validation_years':1,'horizon':5,'purge':5,'lag':lag,'prediction_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'year_order':[2023,2024,2025,2026],'raw_train_window':'testyear-1-trainyears Jan1 to testyear-1 Jan1 (exclusive), purged5/corridor'})
                    outputs.append(pred);summaries.append(old.summarize(pred));k=pred[pred.corridor.eq('KZT')].copy();k.config_id=name+'__KZT';summaries.append(old.summarize(k))
                    print(name,round(time.monotonic()-start,2),flush=True)
    pd.concat(summaries,ignore_index=True).to_csv(HERE/'metrics.csv',index=False)
    pd.concat(outputs,ignore_index=True).to_csv(HERE/'all_predictions.csv.gz',index=False)
    (HERE/'model_receipts.json').write_text(json.dumps(receipts,indent=2))
if __name__=='__main__':run()
