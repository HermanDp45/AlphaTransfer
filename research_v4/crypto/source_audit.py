"""Quality/provenance checks; public API requests only when --download passed."""
from pathlib import Path
import argparse,json,urllib.parse,time
import numpy as np
import pandas as pd
from fetch import request,HERE,DATA,RAW

def download_audit():
    for symbol in ('USDT_RUB','USDT_KZT'):
        for day in ('2020-03-16','2022-03-09','2023-06-24','2025-01-15'):
            start=int(pd.Timestamp(day,tz='UTC').timestamp());end=start+86400-1
            u='https://api.exmo.me/v1.1/candles_history?'+urllib.parse.urlencode({'symbol':symbol,'resolution':60,'from':start,'to':end})
            b,m=request(u,'exmo_hourly_audit')
            if b'rate limit' in b:
                time.sleep(5);request(u,'exmo_hourly_audit',force=True)
            time.sleep(1.2)
    for s in ('usdtrub','btcrub'):
        b,m=request(f'https://api.huobi.pro/market/history/kline?symbol={s}&period=1day&size=2000','htx_rub_history')
        try:d=json.loads(b)
        except Exception:d={}
        if d.get('data'):
            p=pd.DataFrame(d['data']);p['source_date_utc']=pd.to_datetime(p.id,unit='s',utc=True)
            p.to_csv(DATA/f'htx_{s}_audit_only.csv',index=False)
    for group,url in {
        'bybit_p2p_docs':'https://bybit-exchange.github.io/docs/p2p/guide',
        'htx_spot_docs':'https://huobiapi.github.io/docs/spot/v1/en/',
        'exmo_current_orderbook':'https://api.exmo.me/v1.1/order_book?pair=USDT_RUB,USDT_KZT&limit=5',
        'binance_current_orderbook':'https://data-api.binance.vision/api/v3/depth?symbol=USDTKZT&limit=5',
        'exmo_delisting_notice_2026':'https://exmo.me/blog/uvedomleniya/delisting-trading-pairs-on-exmo-me-2',
    }.items():request(url,group)

def audit():
    receipts=[json.loads(p.read_text()) for p in RAW.glob('*.receipt.json')]
    pd.DataFrame(receipts).sort_values(['group','url']).to_csv(HERE/'source_receipts.csv',index=False)
    comparisons=[];bounds=[];parity=[];failures=[]
    for m in receipts:
        b=(HERE/m['raw_path']).read_bytes()
        try:d=json.loads(b)
        except Exception:d={}
        q=urllib.parse.parse_qs(urllib.parse.urlparse(m['url']).query)
        if m['status']!=200 or (isinstance(d,dict) and ('error' in d or d.get('status')=='error' or d.get('retCode',0)!=0)):
            failures.append({**m,'application_error':str(d)[:250] if d else b.decode(errors='replace')[:100]})
        if m['group']=='exmo_me_klines' and '.failed-' not in m['raw_path']:
            rows=d.get('candles',[])
            if rows:
                ts=np.array([r['t'] for r in rows]);bounds.append({'symbol':q['symbol'][0],'year':pd.to_datetime(int(q['from'][0]),unit='s').year,'rows':len(rows),'min_ms':ts.min(),'max_ms':ts.max(),'all_within_requested_seconds':bool(((ts>=int(q['from'][0])*1000)&(ts<=int(q['to'][0])*1000)).all()),'all_midnight_utc':bool((ts%86400000==0).all())})
        if m['group']=='exmo_hourly_audit' and '.failed-' not in m['raw_path']:
            rows=d.get('candles',[]);symbol=q['symbol'][0];day=pd.to_datetime(int(q['from'][0]),unit='s')
            p=pd.DataFrame(rows)
            if p.empty:comparisons.append({'symbol':symbol,'date':day,'hourly_rows':0,'status':'not_available'});continue
            p=p.sort_values('t');daily=pd.read_csv(DATA/f'exmo_me_{symbol}.csv',parse_dates=['source_date'])
            row=daily[daily.source_date.eq(day)].iloc[0]
            agg={'open':p.o.iloc[0],'close':p.c.iloc[-1],'high':p.h.max(),'low':p.l.min(),'volume':p.v.sum()}
            comparisons.append({'symbol':symbol,'date':day,'hourly_rows':len(p),'nonzero_volume_hours':int(p.v.gt(0).sum()),'distinct_hourly_closes':p.c.nunique(),**{k+'_daily_minus_hourly':float(row[k]-v) for k,v in agg.items()}})
        if m['group'].startswith('exmo_parity_'):
            rows=d.get('candles',[])
            if not rows:continue
            p=pd.DataFrame(rows).rename(columns={'t':'open_time','c':'close','v':'volume'})
            raw=pd.read_csv(DATA/'exmo_me_USDT_RUB.csv')
            z=p.merge(raw,on='open_time',suffixes=('_com','_me'))
            parity.append({'year':pd.to_datetime(int(q['from'][0]),unit='s').year,'matched_days':len(z),'max_close_absdiff':abs(z.close_com-z.close_me).max(),'max_volume_absdiff':abs(z.volume_com-z.volume_me).max()})
    pd.DataFrame(comparisons).to_csv(DATA/'exmo_hourly_daily_audit.csv',index=False)
    pd.DataFrame(bounds).to_csv(DATA/'exmo_request_bounds_audit.csv',index=False)
    pd.DataFrame(parity).to_csv(DATA/'exmo_com_me_parity.csv',index=False)
    pd.DataFrame(failures).to_csv(HERE/'failed_endpoints.csv',index=False)
    assert all(b['all_within_requested_seconds'] and b['all_midnight_utc'] for b in bounds)
    # Quantify thin markets, zero returns and temporal scope without assuming
    # that non-zero volume independently proves the absence of wash trades.
    rows=[]
    for symbol in ('USDT_RUB','USDT_KZT'):
        p=pd.read_csv(DATA/f'exmo_me_{symbol}.csv',parse_dates=['source_date'])
        for year,g in p.groupby(p.source_date.dt.year):
            rows.append({'symbol':symbol,'year':year,'bars':len(g),'first':str(g.source_date.min().date()),'last':str(g.source_date.max().date()),'median_close':g.close.median(),'min_close':g.close.min(),'max_close':g.close.max(),'median_base_volume':g.volume.median(),'p10_base_volume':g.volume.quantile(.1),'flat_bars':int(g.high.eq(g.low).sum()),'zero_close_change_share':g.close.diff().eq(0).mean(),'median_range_pct':((g.high/g.low-1)*100).median()})
    pd.DataFrame(rows).to_csv(DATA/'exmo_yearly_quality.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--download',action='store_true');a=ap.parse_args()
    if a.download:download_audit()
    audit()
