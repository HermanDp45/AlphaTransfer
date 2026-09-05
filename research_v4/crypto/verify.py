"""Validate delivered crypto artifacts without network or model retraining."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    checks=[]
    def check(name,condition,detail):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
        assert condition,(name,detail)
    receipts=pd.read_csv(HERE/'source_receipts.csv')
    bad=[r.raw_path for r in receipts.itertuples() if hashlib.sha256((HERE/r.raw_path).read_bytes()).hexdigest()!=r.sha256]
    check('raw source SHA256',not bad,{'receipts':len(receipts),'mismatches':bad})
    q=pd.read_csv(HERE/'data/exmo_request_bounds_audit.csv')
    check('EXMO actual historical bounds',q.all_within_requested_seconds.all() and q.all_midnight_utc.all(),{'nonempty_queries':len(q)})
    q=pd.read_csv(HERE/'data/archive_api_parity.csv');q=q[q.http_status.eq(200)]
    check('Binance ZIP provider checksums and API parity',q.provider_checksum_match.all() and q.max_close_absdiff.eq(0).all() and q.max_volume_absdiff.eq(0).all(),{'archives':len(q),'days':int(q.matched_days.sum())})
    q=pd.read_csv(HERE/'data/exmo_hourly_daily_audit.csv')
    check('EXMO hourly daily OHLC parity',q.filter(regex='(open|close|high|low)_daily_minus').abs().max().max()==0 and q.volume_daily_minus_hourly.abs().max()<1e-8,{'samples':len(q)})
    f=pd.read_parquet(HERE/'data/features_daily.parquet')
    ac=[c for c in f if '_available_date_' in c]
    check('PIT all backward joins',all(not (f[c]>f.date).any() for c in ac),{'availability_columns':len(ac),'dates':len(f)})
    check('unique daily feature rows',not f.date.duplicated().any(),len(f))
    baseline=json.loads((HERE/'output/baseline_verification.json').read_text())
    check('both frozen baselines exactly reproduced',len(baseline)==2 and all(x['max_probability_diff']==0 and x['rows']==4415 and x['candidate_signals_exact'] for x in baseline),baseline)
    configs=json.loads((HERE/'output/configurations.json').read_text())
    files=[HERE/'output'/(c['model']+'_predictions.csv.gz') for c in configs]
    check('all annual configurations complete',all(p.exists() and len(pd.read_csv(p))==4415 for p in files),{'configurations':len(configs),'annual_fits':4*len(configs)})
    m=pd.read_csv(HERE/'output/metrics.csv');dev=m[m.scope.eq('development')]
    base=float(dev[dev.model.eq('basis_short')].brier.iloc[0]);long=float(dev[dev.model.eq('basis_long')].brier.iloc[0])
    check('baseline arithmetic',np.allclose(dev.brier-base,dev.brier_delta_vs_original) and np.allclose(dev.brier-long,dev.brier_delta_vs_long),{'old_brier':base,'long_brier':long})
    (HERE/'output/verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'passed':len(checks),'configurations':len(configs),'annual_fits':4*len(configs)},indent=2))

if __name__=='__main__':main()
