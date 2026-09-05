#!/usr/bin/env python3
"""Official WB research access. Download raw data outside the repository.
Read catalog terms before using --accept-research-terms. Export only aggregates.
"""
import argparse,hashlib,json,re
from datetime import datetime,timezone
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import build_opener,HTTPCookieProcessor,Request

def main():
 p=argparse.ArgumentParser();p.add_argument('--accept-research-terms',action='store_true');p.add_argument('--output',type=Path,default=Path('/private/tmp/at-kg-microdata.zip'));a=p.parse_args()
 if not a.accept_research_terms:p.error('Read https://microdata.worldbank.org/catalog/6523/get-microdata and affirm research terms with --accept-research-terms')
 opener=build_opener(HTTPCookieProcessor(CookieJar()));url='https://microdata.worldbank.org/catalog/6523/get-microdata';body=opener.open(url,timeout=30).read().decode();token=re.search(r'name="ncsrf" value="([^"]+)"',body)
 if not token:raise RuntimeError('Access form changed; inspect official page. Do not bypass access controls.')
 opener.open(Request(url,data=urlencode({'ncsrf':token[1],'accept':'Accept'}).encode(),headers={'Referer':url}),timeout=30).read()
 download='https://microdata.worldbank.org/catalog/6523/download/329993';r=opener.open(download,timeout=60);raw=r.read()
 if not raw.startswith(b'PK'):raise RuntimeError('Expected ZIP, possible access flow change')
 a.output.write_bytes(raw)
 receipt={'download':download,'local_raw_location':str(a.output),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at':datetime.now(timezone.utc).isoformat(),'raw_redistribution':False,'terms':'Research statistical use, confidentiality, citation and no unauthorized copies; source page accepted through normal UI form.'}
 (Path(__file__).parent/'data/microdata_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
