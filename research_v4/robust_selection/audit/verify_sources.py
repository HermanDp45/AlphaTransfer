"""Independent target/maturity and warmup source checks, without fitting."""
from pathlib import Path
import sys,pickle,json,hashlib,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).to_numpy().tobytes()).hexdigest()
def main():
    panel_paths=dict(v3=ROOT/'research_v3/models/panel_extended.pkl',tabm=ROOT/'research_v4/final_sprint/views.pkl')
    base={k:pickle.loads(p.read_bytes()) for k,p in panel_paths.items()};base['tabm']=base['tabm'][0]['2010-01-01',24,1]
    checks=[]
    for branch in ('v3','tabm'):
        p=base[branch]
        raw=pd.read_csv(HERE/branch/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip')
        warm=pd.read_csv(HERE/branch/'warmup.csv.gz',parse_dates=['date'],float_precision='round_trip')
        records=json.loads((HERE/branch/('model_receipts.json' if branch=='v3' else 'receipts.json')).read_text())
        for r in records:
            h=r['train_horizon'];year=r['year'];cid='v3' if branch=='v3' else r['config_id']
            q=p.copy();q['target']=np.nan;q['label_available_date']=q.groupby('corridor').date.shift(-h)
            for _,part in q.groupby('corridor'):
                assert part.date.is_monotonic_increasing
                rates=part.rub_per_unit
                future=pd.concat([rates.shift(-j) for j in range(1,h+1)],axis=1)
                valid=future.notna().all(axis=1)
                target=((future.min(axis=1)/rates-1)*10000+1e-12>=0).astype(float).where(valid)
                q.loc[part.index,'target']=target
                q.loc[part.index,'label_available_date']=part.date.shift(-h)
            cutoff=pd.Timestamp(year,1,1);cs=pd.Timestamp(year-1,1,1);start=cs-pd.DateOffset(months=120)
            tr=q[q.date.ge(start)&q.date.lt(cs)&q.label_available_date.lt(cs)&q.target.notna()].copy()
            if cid=='tabm_kzt':q=q[q.corridor.eq('KZT')];tr=tr[tr.corridor.eq('KZT')]
            cols=r['features'];f=fp(tr[['date','corridor',*cols,'target','label_available_date']])
            expected=r['train_feature_target_sha256'] if branch=='v3' else r['full_train_fingerprint']
            assert f==expected,(branch,cid,h,year,'train fingerprint')
            w=warm[warm.train_horizon.eq(h)&warm.fold_test_year.eq(year)&warm.config_id.eq(cid)]
            before=q[q.date.lt(cs)].groupby('corridor').tail(63)
            key=lambda a:set(zip(a.date,a.corridor))
            assert key(w)==key(before),(branch,cid,h,year,'warmup')
            g=raw[raw.train_horizon.eq(h)&raw.fold_test_year.eq(year)&raw.config_id.eq(cid)&raw.split.eq('test')]
            merged=g.merge(q[['date','corridor','target','label_available_date']],on=['date','corridor'],validate='1:1',suffixes=('_saved','_independent'))
            assert np.array_equal(merged.target_saved,merged.target_independent)
            assert np.array_equal(merged.label_available_date_saved,merged.label_available_date_independent)
            checks.append(dict(status='PASS',branch=branch,config_id=cid,horizon=h,year=year,train_rows=len(tr),test_rows=len(g),train_start=str(tr.date.min()),latest_train_label=str(tr.label_available_date.max()),train_features_target_maturity_fingerprint=f,warmup_is_immediately_prior_63_panel_dates=True,independent_test_target_and_horizon_maturity_exact=True))
    receipt=dict(status='PASS',passed=len(checks),checks=checks,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in panel_paths.values()},audit_sha256=sha(__file__),scope='Independent hand-recomputed future-H event, actual horizon maturity and train-mask fingerprint; no model refit or checkpoint replay.')
    (HERE/'audit/source_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks))))
if __name__=='__main__':main()
