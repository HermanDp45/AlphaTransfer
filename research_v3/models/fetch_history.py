"""Extend CBR history from the first-party XML API; never modify frozen v2 inputs."""
import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request, urllib.parse, xml.etree.ElementTree as ET
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(__file__).resolve().parent/'data'
OUT.mkdir(exist_ok=True)
def get(url,path):
 if path.exists(): return path.read_bytes()
 req=urllib.request.Request(url,headers={'User-Agent':'AlphaTransfer academic research/3.0'})
 b=urllib.request.urlopen(req,timeout=60).read(); path.write_bytes(b); return b
url='https://www.cbr.ru/scripts/XML_daily.asp?date_req=01/09/2026'
r=ET.fromstring(get(url,OUT/'currency_codes.xml'))
codes={x.findtext('CharCode'):x.attrib['ID'] for x in r if x.findtext('CharCode') in ['AMD','KGS','KZT','UZS','TJS']}
def task(item):
 symbol,code=item
 url='https://www.cbr.ru/scripts/XML_dynamic.asp?'+urllib.parse.urlencode({'date_req1':'01/01/2010','date_req2':'31/12/2019','VAL_NM_RQ':code})
 path=OUT/f'cbr_{symbol}_2010_2019.xml'; b=get(url,path)
 rows=[]
 for x in ET.fromstring(b):
  nominal=int(x.findtext('Nominal')); value=float(x.findtext('Value').replace(',','.'))
  rows.append({'date':datetime.datetime.strptime(x.attrib['Date'],'%d.%m.%Y').date().isoformat(),'corridor':symbol,'rub_per_unit':value/nominal,'nominal':nominal,'cbr_value':value})
 print(symbol,len(rows),flush=True)
 return rows,{'url':url,'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(b).hexdigest(),'rows':len(rows)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(task,codes.items()))
old=pd.DataFrame([row for rows,receipt in results for row in rows]); frozen=pd.read_csv(ROOT/'final_solution/data/cbr_daily.csv')
out=pd.concat([old,frozen]).sort_values(['corridor','date']).drop_duplicates(['corridor','date'],keep='last')
out.to_csv(OUT/'cbr_extended.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps({'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_warning':'Latest history; effective dates, not timestamped vintages. V2 rows preserved exactly.','sources':[receipt for rows,receipt in results],'frozen_v2_sha256':hashlib.sha256((ROOT/'final_solution/data/cbr_daily.csv').read_bytes()).hexdigest(),'rows':len(out),'start':out.date.min(),'end':out.date.max()},indent=2))
