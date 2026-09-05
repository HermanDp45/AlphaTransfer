"""Freeze local KZT Sep2/Sep3 extension; prove exact historic FULL33 parity."""
from pathlib import Path
import sys,os,json,hashlib,pickle
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from final_solution.tabm_h3 import features as builder
from final_solution.training import core_experiment as core
HERE=Path(__file__).resolve().parent
RAW_SOURCES={'cbr':'research_v3/models/data/cbr_extended.csv','moex_cny_close':'final_solution/data/normalized/moex_cnyrub_tom.csv','moex_cny_fixing':'final_solution/data/normalized/moex_cny_fixing.csv','oxr':'research_v4/oxr2010_bank/input_oxr_snapshot.csv','halyk':'research_v4/liquidity/halyk_sell_daily.csv','treasury_t10yie':'research_v3/external_data/normalized/treasury_t10yie.csv','treasury_t5yifr':'research_v3/external_data/normalized/treasury_t5yifr.csv'}
TAIL_SOURCE=ROOT/'final_solution/data/normalized/project_cbr_fx_snapshot.csv'
FROZEN=ROOT/'research_v4/final_sprint/views.pkl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).to_numpy().tobytes()).hexdigest()
def main():
    before={str(p.relative_to(ROOT)):sha(p) for p in [*[ROOT/f for f in RAW_SOURCES.values()],TAIL_SOURCE,FROZEN,Path(builder.__file__),Path(core.__file__)]}
    base=pd.read_csv(ROOT/RAW_SOURCES['cbr'],parse_dates=['date'])
    extra=pd.read_csv(TAIL_SOURCE,parse_dates=['effective_date'])
    extra=extra[extra.symbol.eq('KZT')]
    overlap=base[base.corridor.eq('KZT')].merge(extra,left_on='date',right_on='effective_date',validate='1:1')
    difference=(overlap.rub_per_unit-overlap.normalized_value).abs()
    assert (difference<=1e-10).all()
    tail=extra[extra.effective_date.gt(base.date.max())].sort_values('effective_date')
    assert list(tail.effective_date.dt.strftime('%Y-%m-%d'))==['2026-09-02','2026-09-03']
    # Preserve original decimal byte strings. Serializing parsed prefix floats
    # could introduce a second pandas parsing roundoff at quantile boundaries.
    text=(ROOT/RAW_SOURCES['cbr']).read_text().rstrip('\n')+'\n'
    for row in tail.itertuples():
        text+=f'{row.effective_date.date()},KZT,{row.normalized_value},1.0,{row.normalized_value}\n'
    cbrpath=HERE/'latest_cbr.csv';cbrpath.write_text(text)
    paths=dict(RAW_SOURCES,cbr=str(cbrpath.relative_to(ROOT)))
    sources=builder.read_sources(paths,ROOT)
    panel=builder.build_features(sources,'2026-09-03')
    views,_=pickle.loads(FROZEN.read_bytes());frozen=views['2010-01-01',24,1]
    frozen=frozen[frozen.corridor.eq('KZT')].reset_index(drop=True)
    prior=panel[panel.date.le(frozen.date.max())].reset_index(drop=True)
    assert np.array_equal(prior.date,frozen.date)
    assert np.array_equal(prior.session_ordinal,frozen.session_ordinal)
    assert np.array_equal(prior.rub_per_unit,frozen.rub_per_unit)
    a,b=prior[builder.FEATURES].to_numpy(float),frozen[builder.FEATURES].to_numpy(float)
    assert np.array_equal(a,b,equal_nan=True),'FULL33 history changed'
    assert len(panel)==len(frozen)+2 and panel[builder.FEATURES].tail(2).notna().all().all()
    panel=core.add_target(panel,3);panel['label_available_date']=panel.groupby('corridor').date.shift(-3)
    panel['train_horizon']=3
    oldtarget=core.add_target(frozen,3);mature=oldtarget.target.notna()
    for col in ['target','forward_bps','symmetric_bps','regret_bps']:
        assert np.array_equal(panel.iloc[:len(frozen)].loc[mature,col].to_numpy(),oldtarget.loc[mature,col].to_numpy(),equal_nan=True)
    # Before publishing final output, prove every source file stayed unchanged.
    assert all(sha(ROOT/p)==h for p,h in before.items())
    panel.to_pickle(HERE/'latest_panel.pkl')
    panel[['date','corridor','session_ordinal','rub_per_unit',*builder.FEATURES,'target','label_available_date']].tail(8).to_csv(HERE/'latest_tail_audit.csv',index=False)
    (HERE/'source_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
    available=[c for c in panel if c.endswith('_available_date')]+['available_at']
    observation=[c for c in panel if c.endswith('_observation_date')]+['observed_date']
    receipt=dict(status='PASS',scope='KZT only; latest local feature snapshot, distinct from sealed annual test cohorts.',rows=len(panel),date_min=str(panel.date.min()),date_max=str(panel.date.max()),frozen_overlap_rows=len(frozen),feature_count=len(builder.FEATURES),feature_names=builder.FEATURES,old_overlap_features_bitwise_exact=True,old_overlap_rates_and_session_ordinals_exact=True,old_already_mature_H3_outcomes_exact=True,feature_fingerprint=fp(panel[['date','corridor',*builder.FEATURES]]),added_KZT_dates=tail.effective_date.dt.strftime('%Y-%m-%d').tolist(),latest_mature_H3_decision=str(panel.loc[panel.target.notna(),'date'].max()),latest_H3_label_observation=str(panel.label_available_date.max()),source_overlap=dict(rows=len(overlap),max_abs_KZT_rate_difference=float(difference.max()),differences_above_1e_10=int(difference.gt(1e-10).sum())),source_sha256=before,generated_cbr_sha256=sha(cbrpath),latest_panel_sha256=sha(HERE/'latest_panel.pkl'),builder_sha256=sha(builder.__file__),audit_script_sha256=sha(__file__),as_of='2026-09-03 10:05 Europe/Moscow; final deployment cutoff may be later, e.g.2026-09-05',last_two_rows=panel.tail(2)[['date','decision_at',*available,*observation]].to_dict('records'),timing='Inherited recipe: CBR effective-date known10:05MSK proxy; MOEX D+1, OXR max(published,UTC next-midnight)+24h; Halyk D+1 tolerance7 fromavailability; Treasury lag7/maxobservationage14, no pre2020 backfill.',limitations=['Only KZT receives the two new CBR observations; no fabricated updates to other currencies.','Halyk underlying archive ends2026-09-01; later feature joins use that known quote within declared tolerance. It is a market predictor, not an executable Alpha quote.','Treasury/CNY/Halyk histories remain shorter than the official/OXR2010 history; no synthetic early values.','Native honest annual metrics stay frozen on their original mature cohorts. Additional latest labels are solely for finalfit/calibration.'])
    (HERE/'latest_panel_receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','rows','date_max','frozen_overlap_rows','old_overlap_features_bitwise_exact','latest_mature_H3_decision','latest_panel_sha256')}))
if __name__=='__main__':main()
