"""Public Halyk historical SELL quote chart, NOT RUB sender BUY execution.

Frontend explicitly says: 'В графике отображается курс продажи Halyk Bank'.
API paths and DD.MM.YYYY-DD.MM.YYYY parameter come from its own UI code.
"""
from pathlib import Path
from datetime import datetime, timezone
import urllib.request,json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
def main():
    receipts=[];frames=[]
    for client,path in [('personal','exchangerates'),('legal','exchangerates_legal')]:
        for ccy in ('rub','usd','eur'):
            url=f'https://halykbank.kz/api/{path}/{ccy}/01.01.2020-01.09.2026'
            dest=HERE/'raw'/f'halyk_{client}_{ccy}.json'
            if not dest.exists():
                with urllib.request.urlopen(url,timeout=50) as r:dest.write_bytes(r.read())
            data=pd.DataFrame(json.loads(dest.read_bytes()))
            data['date']=pd.to_datetime(data.date_at,utc=True).dt.tz_convert('Asia/Almaty').dt.tz_localize(None).dt.normalize()
            conflicting=data.groupby('date').value.nunique().gt(1)
            # API includes conflicting values with the same midnight timestamp.
            # Their intraday order is not verified; exclude the whole date.
            original=len(data)
            conflict_dates=conflicting[conflicting].index
            data[data.date.isin(conflict_dates)].to_csv(HERE/f'halyk_{client}_{ccy}_conflicts.csv',index=False)
            data=data[~data.date.isin(conflict_dates)].drop_duplicates('date').sort_values('date')
            assert data.date.min()>=pd.Timestamp('2020-01-01') and data.date.max()<=pd.Timestamp('2026-09-01')
            data['client']=client;data['currency']=ccy.upper();data['bank_side']='sell'
            frames.append(data[['date','client','currency','bank_side','value']])
            receipts.append({'url':url,'file':str(dest.relative_to(HERE)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'raw_rows':original,'unique_days':len(data),'min':str(data.date.min().date()),'max':str(data.date.max().date()),'conflicting_duplicate_dates_excluded':len(conflict_dates),'price_side':'bank sell','timestamp_interpretation':'stored midnight Asia/Almaty effective chart date; UTC offset follows IANA historical zone; not verified publication timestamp','availability':'effective chart date +1calendar day; +2 sensitivity','rights':'public download, commercial model/redistribution rights not inferred'})
            print(client,ccy,len(data),flush=True)
    pd.concat(frames,ignore_index=True).to_csv(HERE/'halyk_sell_daily.csv',index=False)
    (HERE/'halyk_receipts.json').write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'source':'https://halykbank.kz/en/exchange-rates','rows':receipts},indent=2))
if __name__=='__main__':main()
