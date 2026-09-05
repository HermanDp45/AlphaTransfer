"""Public, unauthenticated market-data download with immutable source receipts.

Run: python research_v4/crypto/fetch.py
No network or writes at import. Caches HTTP responses including failures.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib, io, json, urllib.request, urllib.error, urllib.parse, zipfile, time
import pandas as pd

HERE=Path(__file__).resolve().parent
RAW=HERE/'raw'; DATA=HERE/'data'
END=pd.Timestamp('2026-09-02',tz='UTC')
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base_volume','taker_quote_volume','ignore']

def request(url,group,force=False):
    RAW.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(url.encode()).hexdigest()[:20]
    path=RAW/f'{group}_{key}.bin'; receipt=path.with_suffix('.receipt.json')
    if path.exists() and receipt.exists():
        if not force:return path.read_bytes(),json.loads(receipt.read_text())
        # Retain failed attempts rather than silently replacing the evidence.
        suffix=f'.failed-{time.time_ns()}'
        failed=RAW/f'{group}_{key}{suffix}.bin'
        meta=json.loads(receipt.read_text());meta['raw_path']=str(failed.relative_to(HERE))
        failed.write_bytes(path.read_bytes());failed.with_suffix('.receipt.json').write_text(json.dumps(meta,indent=2))
    started=datetime.now(timezone.utc).isoformat()
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'AlphaTransfer-research/4.0 (public market historical research)'})
        with urllib.request.urlopen(req,timeout=35) as r:
            b=r.read();status=r.status; final=r.url;ctype=r.headers.get('Content-Type','')
    except urllib.error.HTTPError as e:
        b=e.read();status=e.code; final=e.url;ctype=e.headers.get('Content-Type','')
    except Exception as e:
        b=str(e).encode();status=0;final=url;ctype='error'
    path.write_bytes(b)
    meta={'url':url,'final_url':final,'retrieved_utc':started,'status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content_type':ctype,'raw_path':str(path.relative_to(HERE)),'group':group}
    receipt.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return b,meta

def binance(symbol):
    cursor=int(pd.Timestamp('2017-01-01',tz='UTC').timestamp()*1000); end=int(END.timestamp()*1000)-1
    rows=[]
    for _ in range(6):
        url='https://data-api.binance.vision/api/v3/klines?'+urllib.parse.urlencode({'symbol':symbol,'interval':'1d','startTime':cursor,'endTime':end,'limit':1000})
        b,m=request(url,'binance_klines')
        try:d=json.loads(b)
        except Exception:break
        if m['status']!=200 or not isinstance(d,list) or not d:break
        rows+=d;cursor=int(d[-1][0])+86400000
        if len(d)<1000 or cursor>end:break
    p=pd.DataFrame(rows,columns=COLS)
    if p.empty:return {'provider':'binance','symbol':symbol,'rows':0}
    p=p.apply(pd.to_numeric).drop_duplicates('open_time').sort_values('open_time')
    p['source_date']=pd.to_datetime(p.open_time,unit='ms',utc=True).dt.tz_localize(None)
    p=p[p.source_date<END.tz_localize(None)]
    p.to_csv(DATA/f'binance_{symbol}.csv',index=False)
    return coverage(p,'binance',symbol)

def exmo_chunk(job):
    symbol,year=job
    start=int(pd.Timestamp(f'{year}-01-01',tz='UTC').timestamp())
    end=min(int(pd.Timestamp(f'{year+1}-01-01',tz='UTC').timestamp())-1,int(END.timestamp())-1)
    url='https://api.exmo.me/v1.1/candles_history?'+urllib.parse.urlencode({'symbol':symbol,'resolution':'D','from':start,'to':end})
    b,m=request(url,'exmo_me_klines')
    try:d=json.loads(b)
    except Exception:d={}
    for delay in (3,10,20):
        if isinstance(d,dict) and 'rate limit' in str(d.get('error','')).lower():
            time.sleep(delay)
            b,m=request(url,'exmo_me_klines',force=True)
            try:d=json.loads(b)
            except Exception:d={}
        else:break
    return symbol,year,d.get('candles',[]) if isinstance(d,dict) else []

def coverage(p,provider,symbol):
    return {'provider':provider,'symbol':symbol,'rows':len(p),'first_date':str(p.source_date.min().date()),'last_date':str(p.source_date.max().date()),'zero_volume_rows':int(p.volume.le(0).sum()),'duplicate_dates':int(p.source_date.duplicated().sum()),'calendar_days_absent':int((p.source_date.max()-p.source_date.min()).days+1-len(p))}

def documents_and_probes():
    docs={
      'binance_publicdata_readme':'https://raw.githubusercontent.com/binance/binance-public-data/master/README.md',
      'binance_marketdata_docs':'https://developers.binance.com/en/docs/products/spot/faqs/market_data_only',
      'binance_terms_pdf':'https://bin.bnbstatic.com/static/cms/cg08ou2ak0tn7mcplvfg/file/bf4879710c904b991848972ec4818ba2cf9e4ce314c09adae84fa2750d3477f7.pdf',
      'binance_p2p_rub_ceased':'https://www.binance.com/en/support/announcement/detail/3016096ace174be381fa22b6636e2c5f',
      'exmo_api_docs':'https://www.postman.com/exmo-finance/exmo-finance-s-public-workspace/collection/uhjz2oi/exmo-me-api',
      'exmo_terms':'https://exmo.me/blog/user-agreement',
      'exmo_com_split':'https://exmo.com/blog/en/product/exmo-coms-product-release-recap-main-updates-of-the-year',
      'exmo_me_split':'https://exmo.me/blog/uvedomleniya/exmo-2022-rewind',
      'binance_metadata':'https://data-api.binance.vision/api/v3/exchangeInfo?symbols=%5B%22USDTRUB%22,%22BTCRUB%22,%22BTCUSDT%22,%22USDTKZT%22,%22USDCUSDT%22%5D',
      'binance_kz_blocked':'https://api.binance.kz/api/v3/klines?symbol=USDTKZT&interval=1d&limit=3',
      'exmo_me_metadata':'https://api.exmo.me/v1.1/pair_settings',
      'exmo_com_metadata':'https://api.exmo.com/v1.1/pair_settings',
      'exmo_parity_2020':'https://api.exmo.com/v1.1/candles_history?symbol=USDT_RUB&resolution=D&from=1577836800&to=1609459199',
      'exmo_parity_2023':'https://api.exmo.com/v1.1/candles_history?symbol=USDT_RUB&resolution=D&from=1672531200&to=1704067199',
      'bybit_metadata':'https://api.bybit.com/v5/market/instruments-info?category=spot',
      'bybit_kzt_probe':'https://api.bybit.com/v5/market/kline?category=spot&symbol=USDTKZT&interval=D&limit=3',
      'bybit_rub_probe':'https://api.bybit.com/v5/market/kline?category=spot&symbol=USDTRUB&interval=D&limit=3',
      'htx_metadata':'https://api.huobi.pro/v1/common/symbols',
      'htx_kzt_probe':'https://api.huobi.pro/market/history/kline?symbol=usdtkzt&period=1day&size=3',
      'ataix_public_metadata':'https://api.ataix.kz/api/cmc/v1/overview',
      'ataix_public_docs':'https://stage-api.ataix.kz/api/docs/cmc/',
      'currency_com_metadata':'https://api-adapter.backend.currency.com/api/v2/exchangeInfo',
    }
    with ThreadPoolExecutor(4) as pool:
        for name,result in zip(docs,pool.map(lambda item:request(item[1],item[0]),docs.items())):
            print('probe',name,result[1]['status'],flush=True)
    # A small independent archive/API parity audit, including µs-era files.
    pairs=[('USDTRUB','2020-01'),('USDTRUB','2023-01'),('USDTRUB','2024-01'),('USDTRUB','2025-01'),('BTCUSDT','2025-01'),('USDTKZT','2025-01'),('USDTKZT','2026-05')]
    audit=[]
    for symbol,month in pairs:
        u=f'https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip'
        b,m=request(u,'binance_archive')
        rec={'symbol':symbol,'month':month,'http_status':m['status'],'sha256':m['sha256']}
        if m['status']==200:
            checksum,cm=request(u+'.CHECKSUM','binance_checksum')
            rec['provider_checksum_match']=cm['status']==200 and m['sha256']==checksum.decode().split()[0]
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                p=pd.read_csv(z.open(z.namelist()[0]),header=None,names=COLS)
            p['source_date']=pd.to_datetime(p.open_time,unit='us' if p.open_time.max()>1e14 else 'ms',utc=True).dt.tz_localize(None)
            q=pd.read_csv(DATA/f'binance_{symbol}.csv',parse_dates=['source_date'])
            matched=p.merge(q,on='source_date',suffixes=('_archive','_api'))
            rec['matched_days']=len(matched)
            rec['max_close_absdiff']=(matched.close_archive-matched.close_api).abs().max()
            rec['max_volume_absdiff']=(matched.volume_archive-matched.volume_api).abs().max()
        audit.append(rec)
    pd.DataFrame(audit).to_csv(DATA/'archive_api_parity.csv',index=False)

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    reports=[]
    for s in ('USDTRUB','BTCRUB','BTCUSDT','USDCUSDT','USDTKZT'):
        r=binance(s);reports.append(r);print(r,flush=True)
    symbols=['USDT_RUB','USDT_KZT','BTC_RUB','BTC_USDT','USDT_USD']
    chunks={s:[] for s in symbols}
    # Candle history has a tighter burst limit than exchange metadata. Respect
    # the rate limit with sequential requests and backoff, never rotate hosts.
    for s in symbols:
        for y in range(2017,2027):
            ss,yy,rows=exmo_chunk((s,y));chunks[ss]+=rows
            time.sleep(1.1)
    for s,rows in chunks.items():
        p=pd.DataFrame(rows).rename(columns={'t':'open_time','o':'open','c':'close','h':'high','l':'low','v':'volume'})
        if p.empty:reports.append({'provider':'exmo_me','symbol':s,'rows':0});continue
        p=p.apply(pd.to_numeric).drop_duplicates('open_time').sort_values('open_time')
        p['source_date']=pd.to_datetime(p.open_time,unit='ms',utc=True).dt.tz_localize(None)
        p=p[p.source_date<END.tz_localize(None)]
        p.to_csv(DATA/f'exmo_me_{s}.csv',index=False)
        r=coverage(p,'exmo_me',s);reports.append(r);print(r,flush=True)
    pd.DataFrame(reports).to_csv(DATA/'coverage.csv',index=False)
    documents_and_probes()
    receipts=[json.loads(p.read_text()) for p in RAW.glob('*.receipt.json')]
    pd.DataFrame(receipts).sort_values(['group','url']).to_csv(HERE/'source_receipts.csv',index=False)
    print('DONE',len(receipts),'source receipts',flush=True)

if __name__=='__main__': main()
