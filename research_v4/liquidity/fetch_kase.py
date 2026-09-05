"""Fetch public KASE website daily spot results, with resumable receipts.

Uses the same endpoint/date_trade parameter as KASE's public frontend.
No credentials, paid feed, private endpoint or fabricated historical quotes.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, date, timedelta
import urllib.request, urllib.error, json, time, hashlib

HERE=Path(__file__).resolve().parent
RAW=HERE/'raw'/'kase_spot'
RAW.mkdir(parents=True,exist_ok=True)
def one(day):
    path=RAW/f'{day}.json'
    url=f'https://kase.kz/api/trade-results/currency-spot?date_trade={day}'
    if not path.exists():
        for attempt in range(4):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'AlphaTransfer research; public historical data'})
                with urllib.request.urlopen(request,timeout=40) as response:raw=response.read()
                rows=json.loads(raw)
                if not isinstance(rows,list):raise ValueError('unexpected response')
                if any(r['date_trade'][:10]!=day for r in rows):raise ValueError('API returned a different historical date')
                path.write_bytes(raw)
                break
            except Exception as exc:
                if attempt==3:return {'day':day,'url':url,'error':repr(exc)}
                time.sleep(1+attempt*2)
    raw=path.read_bytes();rows=json.loads(raw)
    assert all(r['date_trade'][:10]==day for r in rows)
    return {'day':day,'url':url,'rows':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'file':str(path.relative_to(HERE))}
def main():
    days=[];day=date(2020,1,1)
    while day<=date(2026,9,1):
        days.append(day.isoformat())
        day+=timedelta(days=1)
    receipts=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,d) for d in days]
        for i,future in enumerate(as_completed(futures),1):
            receipts.append(future.result())
            if i%100==0: print(i,'/',len(days),'errors',sum('error' in r for r in receipts),flush=True)
    receipts.sort(key=lambda r:r['day'])
    (HERE/'kase_receipts.json').write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'source':'KASE public trade-results API','access':'publicly downloadable; no assumption of unrestricted ML or redistribution rights','cutoff':'end-of-day snapshot, decision date >= trade date + 1 calendar day; sensitivity +2','weekdays_only':False,'rows':receipts},indent=2))
    rows=[r for p in sorted(RAW.glob('*.json')) for r in json.loads(p.read_text())]
    import pandas as pd
    pd.DataFrame(rows).to_csv(HERE/'kase_spot_daily.csv',index=False)
    print('complete',len(rows),'source rows',flush=True)
if __name__=='__main__':main()
