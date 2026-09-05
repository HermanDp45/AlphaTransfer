#!/usr/bin/env python3
"""Post-audit numerical attribution: exact common OXR features after2018-09-01."""
import datetime
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('foundation_canonical_fusion',HERE/'experiment.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
OUT=HERE/'canonical';START=pd.Timestamp('2018-09-01')
ARMS=['synth_basis2010_D2','small_basis2010_D2','synth_kzt_basis2010_halyk_H1O2']


def main():
    OUT.mkdir(exist_ok=True)
    protocol=OUT/'protocol.json'
    assert not protocol.exists(),'This numerical control is a single frozen run'
    fusion.save(protocol,{'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'post-audit numerical attribution after original test readout; not independent new validation','arms':ARMS,'cutoffs':['2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'],'models':15,'common_feature_rule':'For dates>=2018-09-01 replace all numerical OXR features with exact same-snapshot since2018-06-17 view; preserve every earlier feature value from2010 view. No other feature or label changes.','common_columns':fusion.oxr.RET+fusion.oxr.BASIS+fusion.oxr.COVER,'new_hyperparameters':False,'new_selection':False,'original_selection_preserved':True,'source':'same frozen root input_oxr_snapshot.csv for both views','head':'same fixed10year HGB and forecast cache, prior12month calibration and matured labels, explicitJan/Mar history replay','original_protocol_sha256':fusion.sha(HERE/'protocol.json'),'code_sha256':fusion.sha(__file__)})
    fusion.OUT=OUT
    builder=fusion.FeatureBuilder();forecasts=[];checks=[]
    short=fusion.oxr.build_panel(delay=24,since='2018-06-17',raw=builder.raw,extended=True)
    columns=fusion.oxr.RET+fusion.oxr.BASIS+fusion.oxr.COVER
    short=short.set_index(['date','corridor'])
    configurations={s['name']:s for s in fusion.specs()}
    with threadpool_limits(limits=1):
        for original in ARMS:
            for cutoff in ['2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01']:
                year=pd.Timestamp(cutoff).year;s=dict(configurations[original]);s['name']=original+'_canonical';s['original_config_id']=original;s['canonical_common_since']=str(START.date())
                p,features,cache=builder.build(s,year)
                before=p[features].copy();common=p.date.ge(START)
                keys=pd.MultiIndex.from_frame(p.loc[common,['date','corridor']])
                original_common=p.loc[common,columns].copy()
                p.loc[common,columns]=short.loc[keys,columns].to_numpy()
                assert np.array_equal(p.loc[common,columns].to_numpy(),short.loc[keys,columns].to_numpy(),equal_nan=True)
                assert np.array_equal(p.loc[~common,features].to_numpy(),before.loc[~common,features].to_numpy(),equal_nan=True)
                others=[f for f in features if f not in columns]
                assert np.array_equal(p[others].to_numpy(),before[others].to_numpy(),equal_nan=True)
                pred=fusion.evaluate(p,features,cache,s,cutoff)
                name=s['name']+'_'+cutoff;path=OUT/'output'/f'{name}_head.joblib'
                pack=joblib.load(path)
                test=p.merge(pred[['date','corridor','raw_probability','probability']],on=['date','corridor'],validate='one_to_one')
                raw=pack['model'].predict_proba(test[features+['corridor']])[:,1]
                prob=fusion.core.apply_platt(pack['calibrator'],raw)
                error=float(abs(prob-test.probability).max());assert error<1e-12
                receipt=json.loads((OUT/'output'/f'{name}_receipt.json').read_text());assert all(receipt['checks'].values())
                deltas={c:float(np.nanmax(abs(original_common[c].to_numpy()-p.loc[common,c].to_numpy()))) for c in columns}
                checks.append({'config_id':s['name'],'cutoff':cutoff,'common_OXR_values_exactly_match2018_view':True,'earlier_features_bitwise_preserved':True,'all_non_OXR_features_bitwise_preserved':True,'max_abs_common_feature_changes':deltas,'saved_probability_reconstruction_max_abs_diff':error,'head_weights_sha256':fusion.sha(path),'purged_label_maturity_pass':True})
                forecasts.append(pred)
    pred=pd.concat(forecasts,ignore_index=True);pred.to_csv(OUT/'predictions.csv.gz',index=False)
    new=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['date']);old=pd.read_csv(HERE/'predictions.csv.gz',parse_dates=['date'],low_memory=False)
    data=pd.concat([new,old],ignore_index=True)
    rows=[];intervals=[]
    stages={'development_2023_2025':data.cutoff.lt('2026-01-01'),'jan2026_retrospective':data.cutoff.eq('2026-01-01'),'march2026_retrospective':data.cutoff.eq('2026-03-01')}
    for stage,mask in stages.items():
        for scope in ['all','KZT']:
            scoped=data[mask]
            if scope=='KZT':scoped=scoped[scoped.corridor.eq('KZT')]
            grouped={cid:f.sort_values(['date','corridor']).reset_index(drop=True) for cid,f in scoped.groupby('config_id') if not(scope=='all' and cid.startswith('synth_kzt_'))}
            for original in ARMS:
                cid=original+'_canonical'
                if cid not in grouped:continue
                f=grouped[cid]
                rows.append({'stage':stage,'scope':scope,'config_id':cid,**fusion.score(f)})
                references=[original,'v3_long']
                if original=='synth_basis2010_D2':references+=['synth_basis2018_D2','synth_no_oxr']
                if original=='small_basis2010_D2':references+=['small_no_oxr']
                if original.startswith('synth_kzt_'):references+=['synth_kzt_halyk_D1','synth_kzt_no_oxr']
                for baseline in references:
                    with threadpool_limits(limits=1):result=fusion.paired(f,grouped[baseline],10000)
                    intervals.append({'stage':stage,'scope':scope,'config_id':cid,'benchmark':baseline,**result})
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    fusion.save(OUT/'verification.json',{'status':'PASS','models_reconstructed':15,'new_neural_fits':0,'new_neural_forecasts':0,'canonicalization_checks':checks,'only_declared_common_OXR_values_changed':True,'all_same_date_outcomes_paired':True,'original_selection_sha256':fusion.sha(HERE/'selection.json'),'source_snapshot_sha256':fusion.sha(fusion.SNAPSHOT),'protocol_sha256':fusion.sha(protocol),'code_sha256':fusion.sha(__file__),'interpretation':'post-audit numerical attribution, all predetermined configurations retained; no2026model selection'})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
