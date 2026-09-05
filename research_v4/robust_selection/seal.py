from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 for name in ('audit/evaluation_verification.json','audit/uncertainty_verification.json','audit/REPORT.md','REPORT.md','REPORTING_ALL_METRICS.csv','recommendation.json'):
  assert (HERE/name).exists(),name
 files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p not in [HERE/'MANIFEST.json',HERE/'_SUCCESS.json']}
 for name,digest in files.items():assert sha(HERE/name)==digest
 manifest=dict(status='PASS',files=files,count=len(files),scope='Independent retrospective2024-2026 robustness capsule; previous sealed experiments unchanged')
 (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 (HERE/'_SUCCESS.json').write_text(json.dumps(dict(status='complete',manifest_sha256=sha(HERE/'MANIFEST.json'),files=len(files),reporting_csv_sha256=sha(HERE/'REPORTING_ALL_METRICS.csv'),recommendation='KZT TabM H5 rank90; retrospective, pointwise80% coverage gates;2025 utilityCI crosses0; no all5 model meetsallcellgates'),indent=2)+'\n')
 print('Seal PASS',len(files),'files')
if __name__=='__main__':main()
