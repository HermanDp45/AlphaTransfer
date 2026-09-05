"""Readable KASE duplicate-resolution ledger and independent XLS cross-check."""
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as E
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    raw=pd.read_csv(HERE/'kase_spot_daily.csv')
    raw['completeness']=(raw.average.gt(0)&raw.volume.gt(0)&raw.deals.gt(0)).astype(int)*2+(raw.high.gt(0)&raw.low.gt(0)).astype(int)
    selected=raw.sort_values(['date_trade','code','completeness','num_sess'],na_position='first').drop_duplicates(['date_trade','code'],keep='last')
    duplicates=raw[raw.duplicated(['date_trade','code'],keep=False)].copy()
    duplicates['selected']=duplicates.index.isin(selected.index)
    duplicates['resolution']='active price/volume/deals; then positive range; then latest session; do not sum'
    duplicates.to_csv(HERE/'kase_session_resolution.csv',index=False)
    assert duplicates.groupby(['date_trade','code']).selected.sum().eq(1).all()
    path=HERE/'raw/kase_usdkzt_archive.xlsx';z=zipfile.ZipFile(path)
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    strings=[''.join(si.itertext()) for si in E.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
    rows=[]
    for row in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',ns)[2:]:
        values=[]
        for cell in row:
            v=cell.find('s:v',ns);value=v.text if v is not None else ''.join(cell.itertext())
            values.append(strings[int(value)] if cell.get('t')=='s' else value)
        if len(values)>=4:rows.append({'date':values[0],'xls_1100':values[1],'xls_1530':values[2],'xls_1700':values[3]})
    xls=pd.DataFrame(rows);xls['date']=pd.to_datetime(xls.date,format='%d.%m.%Y',errors='coerce');xls.xls_1700=pd.to_numeric(xls.xls_1700,errors='coerce')
    usd=selected[selected.code.eq('USDKZT_TOM')].copy();usd['date']=pd.to_datetime(usd.date_trade).dt.normalize()
    comparison=usd.merge(xls[['date','xls_1700']],on='date',how='inner',validate='one_to_one');comparison['difference']=comparison.average-comparison.xls_1700
    comparison[['date','average','xls_1700','difference','volume','deals','num_sess','completeness']].to_csv(HERE/'kase_usd_xls_crosscheck.csv',index=False)
    finite=comparison.difference.notna()
    details={'KASE_duplicate_date_instrument_cells':len(duplicates.groupby(['date_trade','code'])),'exactly_one_chosen_per_duplicate_cell':True,'XLS_independent_export_url':'https://kase.kz/api/indicators/usd-kzt-tom/archive-xls?start_date=2020-01-01&end_date=2026-09-01','XLS_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'XLS_matched_active_values':int(finite.sum()),'XLS_exact_to_1e_8':int(comparison.loc[finite,'difference'].abs().le(1e-8).sum()),'XLS_max_absolute_difference':float(comparison.loc[finite,'difference'].abs().max()),'XLS_mean_absolute_difference':float(comparison.loc[finite,'difference'].abs().mean()),'nonmatching_values_are_retained_for_audit':True,'chosen_incomplete_session_cases':{'2020-03-31':int(usd.loc[usd.date.eq('2020-03-31'),'deals'].iloc[0]),'2020-08-21':int(usd.loc[usd.date.eq('2020-08-21'),'deals'].iloc[0])},'history_days_with_records':int(raw.date_trade.nunique()),'raw_session_rows':len(raw),'calendar_requests':len(json.loads((HERE/'kase_receipts.json').read_text())['rows'])}
    (HERE/'source_audit.json').write_text(json.dumps(details,indent=2))
    print(json.dumps(details,indent=2))
if __name__=='__main__':main()
