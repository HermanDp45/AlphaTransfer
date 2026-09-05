"""Download public primary remittance reports and record exact bytes/provenance."""
from pathlib import Path
from urllib.request import urlopen, Request
import json,hashlib,datetime
ROOT=Path(__file__).resolve().parent/'data/raw'
SOURCES=[('iom_kg_return_2024.pdf','https://kyrgyzstan.iom.int/sites/g/files/tmzbdl1321/files/documents/2024-11/kg_return-migrant-survey_r3_eng.pdf'),('iom_kg_return_2023.pdf','https://kyrgyzstan.iom.int/sites/g/files/tmzbdl1321/files/documents/2024-07/kg_baseline-and-returning-migrant-worker-survey.pdf'),('wb_rpp_baseline2016.pdf','https://documents1.worldbank.org/curated/en/552541540823620142/pdf/131455-RPPbaselinesurveyFINAL.pdf'),('kg_microdata_access.html','https://microdata.worldbank.org/catalog/6523/get-microdata')]

def main():
 ROOT.mkdir(parents=True,exist_ok=True);logs=[]
 for name,url in SOURCES:
  rec={'name':name,'url':url,'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  try:
   r=urlopen(Request(url,headers={'User-Agent':'AlphaTransfer research/1.0'}),timeout=30);data=r.read()
   if name.endswith('.pdf') and not data.startswith(b'%PDF'):raise ValueError('Expected PDF bytes')
   (ROOT/name).write_bytes(data)
   rec.update(status=r.status,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  except Exception as e:rec['error']=str(e)
  print(rec,flush=True);logs.append(rec)
 (ROOT/'download_manifest.json').write_text(json.dumps(logs,indent=2)+'\n')
if __name__=='__main__':main()
