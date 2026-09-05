"""Causal FULL33 features from normalized daily source tables.

Source formulas/float parsing intentionally reproduce the frozen training recipe.
No labels, future returns, research imports, network calls or implicit backfill.
"""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import json
import numpy as np
import pandas as pd

BASE=[*(f'ret{n}' for n in (1,3,5,10,20,60)),*(f'pr{n}' for n in (20,60,120,252)),*(f'vol{n}' for n in (5,20,60)),'volratio','moex_cny_close_minus_fixing_same_session']
OXR=['oxr_log_basis','oxr_basis_chg1','oxr_basis_chg5','oxr_basis_z20','oxr_available','oxr_age_days']
HALYK=['halyk_rub_sell_official_basis','halyk_rub_personal_legal_gap','halyk_personal_rub_ret1','halyk_personal_rub_ret5','halyk_personal_usd_ret1','halyk_personal_usd_ret5']
TREASURY=[f'treasury_{symbol}_stale_{suffix}' for symbol in ('t10yie','t5yifr') for suffix in ('level','chg5','chg20')]
FEATURES=BASE+OXR+HALYK+TREASURY
SOURCE_NAMES=('cbr','moex_cny_close','moex_cny_fixing','oxr','halyk','treasury_t10yie','treasury_t5yifr')


def read_sources(config:Mapping[str,str],base:Path|None=None)->dict[str,pd.DataFrame]:
    """Load explicit paths; caller decides whether they are frozen bundle inputs."""
    if set(config)!=set(SOURCE_NAMES):raise ValueError('Source map must contain exactly '+', '.join(SOURCE_NAMES))
    # The original research recipe used pandas ordinary float parsing. Changing
    # this to round_trip changes quantile-boundary scores on some exact ties.
    return {name:pd.read_csv((base or Path())/Path(config[name])) for name in SOURCE_NAMES}


def _dates(frame,column):
    dates=pd.to_datetime(frame[column],errors='raise')
    if dates.dt.tz is not None:raise ValueError(f'{column} must be a naive calendar date')
    if not dates.eq(dates.dt.normalize()).all():raise ValueError(f'{column} contains non-midnight timestamps')
    return dates


def _unique(frame,keys,source):
    if frame.duplicated(keys).any():raise ValueError(f'{source}: duplicate observation keys {keys}')


def _join(panel,table,prefix,lag,max_age,*,age_from_observation):
    right=table.copy().sort_values('observation_date')
    right['available_date']=right.observation_date+pd.Timedelta(days=lag)
    right=right.rename(columns={c:prefix+c for c in right if c!='available_date'})
    joined=pd.merge_asof(panel.sort_values('date'),right.sort_values('available_date'),left_on='date',right_on='available_date',direction='backward',tolerance=None if age_from_observation else pd.Timedelta(days=max_age))
    joined=joined.rename(columns={'available_date':prefix+'available_date'})
    if age_from_observation:
        stale=(joined.date-joined[prefix+'observation_date']).dt.days>max_age
        fields=[prefix+c for c in table if c!='observation_date'];joined.loc[stale,fields]=np.nan
    known=joined[prefix+'available_date'].notna()
    if (joined.loc[known,prefix+'available_date']>joined.loc[known,'date']).any():raise ValueError('Future source crossed join')
    return joined


def build_features(sources:Mapping[str,pd.DataFrame],as_of:str|pd.Timestamp,*,corridor='KZT')->pd.DataFrame:
    """Build the entire observed CBR prefix ending at as_of, then select KZT.

    CBR date is the effective decision date, assumed known at10:05 Moscow. Raw
    daily publication timestamps are unavailable for CBR/MOEX/Halyk/Treasury;
    their inherited conservative lags are explicit proxy timing assumptions.
    """
    if set(sources)!=set(SOURCE_NAMES):raise ValueError('Incomplete source tables')
    if corridor!='KZT':raise ValueError('This profile is validated only for KZT')
    end=pd.Timestamp(as_of)
    if end.tz is not None:end=end.tz_convert('Europe/Moscow').tz_localize(None)
    end=end.normalize()
    p=sources['cbr'].copy();p['date']=_dates(p,'date');p=p[p.date.le(end)].copy()
    _unique(p,['date','corridor'],'CBR')
    if p.empty or not p.corridor.eq('KZT').any():raise ValueError('No KZT source history at as_of')
    if not np.isfinite(p.rub_per_unit).all() or not p.rub_per_unit.gt(0).all():raise ValueError('CBR rates must be finite positive numbers')
    # Match all-corridor historical assembly before the final KZT extraction.
    parts=[]
    for _,g in p.sort_values(['corridor','date']).groupby('corridor'):
        g=g.copy();log=np.log(g.rub_per_unit)
        for n in (1,3,5,10,20,60):g[f'ret{n}']=log.diff(n)
        for n in (5,20,60):g[f'vol{n}']=g.ret1.rolling(n).std()
        for n in (20,60,120,252):
            g[f'pr{n}']=g.rub_per_unit.rolling(n,min_periods=n).apply(lambda v:float((np.sum(v<v[-1])+.5*np.sum(np.isclose(v,v[-1])))/len(v)),raw=True)
        g['volratio']=g.vol5/g.vol60;g['session_ordinal']=np.arange(len(g));parts.append(g)
    p=pd.concat(parts).sort_values(['date','corridor']).reset_index(drop=True)
    for name,prefix in [('moex_cny_close','cny_close_'),('moex_cny_fixing','cny_fix_')]:
        f=sources[name].copy();f['observation_date']=_dates(f,'TRADEDATE');f=f[f.observation_date.le(end)].sort_values('observation_date')
        _unique(f,['observation_date'],name);col='CLOSE_RUB_PER_UNIT' if 'CLOSE_RUB_PER_UNIT' in f else 'CLOSE'
        value=pd.to_numeric(f[col],errors='coerce');f=f[value.gt(0)].copy();f['log_level']=np.log(value.loc[f.index])
        p=_join(p,f[['observation_date','log_level']],prefix,1,7,age_from_observation=True)
    p[BASE[-1]]=(p.cny_close_log_level-p.cny_fix_log_level).where(p.cny_close_observation_date.eq(p.cny_fix_observation_date))
    f=sources['oxr'].copy();f['observed_date']=_dates(f,'date');f=f[f.observed_date.le(end)].copy()
    f['published']=pd.to_datetime(f.published_at_utc,utc=True,errors='raise')
    dayend=f.observed_date.dt.tz_localize('UTC')+pd.Timedelta(days=1)
    f['available_at']=pd.concat([f.published,dayend],axis=1).max(axis=1)+pd.Timedelta(hours=24)
    oxr_parts=[]
    for c,left in p.groupby('corridor'):
        r=f[f.quote.eq(c)].sort_values('observed_date').copy();_unique(r,['observed_date'],'OXR '+c)
        if not r.rub_per_quote.gt(0).all():raise ValueError('OXR rate must be positive')
        r['oxr_log_rate']=np.log(r.rub_per_quote)
        left=left.sort_values('date').copy();left['decision_at']=(left.date.dt.tz_localize('Europe/Moscow')+pd.Timedelta(hours=10,minutes=5)).dt.tz_convert('UTC')
        joined=pd.merge_asof(left,r[['available_at','published','observed_date','oxr_log_rate']].sort_values('available_at'),left_on='decision_at',right_on='available_at',direction='backward',tolerance=pd.Timedelta(days=7))
        joined['oxr_available']=joined.available_at.notna().astype(float);joined['oxr_age_days']=(joined.decision_at-joined.published).dt.total_seconds()/86400
        b=joined.oxr_log_rate-np.log(joined.rub_per_unit);joined['oxr_log_basis']=b
        joined['oxr_basis_chg1']=b.diff();joined['oxr_basis_chg5']=b.diff(5);joined['oxr_basis_z20']=(b-b.rolling(20,min_periods=10).mean())/b.rolling(20,min_periods=10).std().clip(lower=1e-6)
        oxr_parts.append(joined)
    p=pd.concat(oxr_parts,ignore_index=True).sort_values(['date','corridor']).reset_index(drop=True)
    knownkzt=p[p.corridor.eq('KZT')].set_index('date').rub_per_unit;p['known_kzt_rub_per_unit']=p.date.map(knownkzt)
    bank=sources['halyk'].copy();bank['date']=_dates(bank,'date');bank=bank[bank.date.le(end)].copy()
    if 'bank_side' not in bank or not bank.bank_side.eq('sell').all():raise ValueError('Halyk input must explicitly be BANK SELL quotes')
    _unique(bank,['date','client','currency'],'Halyk')
    for client,ccy in [('personal','RUB'),('legal','RUB'),('personal','USD')]:
        b=bank[bank.client.eq(client)&bank.currency.eq(ccy)].sort_values('date');log=np.log(b.value.where(b.value>0))
        table=pd.DataFrame({'observation_date':b.date,'log_price':log,'ret1':log.diff(),'ret5':log.diff(5)})
        p=_join(p,table,f'halyk_{client}_{ccy.lower()}_',1,7,age_from_observation=False)
    p[HALYK[0]]=p.halyk_personal_rub_log_price+np.log(p.known_kzt_rub_per_unit)
    p[HALYK[1]]=(p.halyk_personal_rub_log_price-p.halyk_legal_rub_log_price).where(p.halyk_personal_rub_observation_date.eq(p.halyk_legal_rub_observation_date))
    for symbol in ('t10yie','t5yifr'):
        sid='treasury_'+symbol;f=sources[sid].copy();f['observation_date']=_dates(f,'source_date');f=f[f.observation_date.le(end)].sort_values('observation_date');_unique(f,['observation_date'],sid)
        v=f[sid];table=pd.DataFrame({'observation_date':f.observation_date,'level':v,'chg5':v.diff(5),'chg20':v.diff(20)})
        p=_join(p,table,sid+'_stale_',7,14,age_from_observation=True)
        p.loc[p.date.lt('2020-01-01'),[sid+'_stale_'+s for s in ('level','chg5','chg20')]]=np.nan
    p=p.sort_values(['date','corridor']).reset_index(drop=True)
    q=p[p.corridor.eq(corridor)].copy()
    q['feature_known_at']=q.decision_at;q['feature_date']=q.date
    q['source_missing_features']=q[FEATURES].isna().sum(axis=1)
    if np.isinf(q[FEATURES].to_numpy(float)).any():raise ValueError('Infinite engineered feature')
    keep=['date','feature_date','feature_known_at','decision_at','corridor','session_ordinal','rub_per_unit',*FEATURES,'source_missing_features','available_at','published','observed_date',*[c for c in q if c.endswith('_available_date') or c.endswith('_observation_date')]]
    return q[list(dict.fromkeys(keep))].reset_index(drop=True)
