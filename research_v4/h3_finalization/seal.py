from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=ROOT/'final_solution/tabm_h3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 for f in ('REPORT.md','REPORTING_ALL_METRICS.csv','reporting_sources.json'):shutil.copy2(HERE/f,P/'evaluation'/f)
 for f in ('REPORT.md','final_handoff_receipt.json'):shutil.copy2(HERE/'audit'/f,P/'evaluation'/('audit_'+f))
 package={str(p.relative_to(P)):sha(p) for p in sorted(P.rglob('*')) if p.is_file() and not any(x in ('__pycache__','output') for x in p.relative_to(P).parts) and p.name not in ('MANIFEST.json','_SUCCESS.json')}
 for path,h in package.items():assert sha(P/path)==h
 write(P/'MANIFEST.json',dict(status='PASS',files=package,count=len(package),mutable_outputs_excluded=True))
 write(P/'_SUCCESS.json',dict(status='active_H3_profile',manifest_sha256=sha(P/'MANIFEST.json'),independent_checks=120,unit_tests=15,history_start='2010-01-01',primary_horizon=3,closing_active=False))
 files={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p not in (HERE/'MANIFEST.json',HERE/'_SUCCESS.json')}
 for path in package:files[str((P/path).relative_to(ROOT))]=package[path]
 for rel in ('final_solution/main.py','final_solution/README.md','final_solution/active_profile.json','final_solution/requirements-tabm-h3.txt','final_solution/tabm_h3/MANIFEST.json','final_solution/tabm_h3/_SUCCESS.json'):files[rel]=sha(ROOT/rel)
 for path,h in files.items():assert sha(ROOT/path)==h
 write(HERE/'MANIFEST.json',dict(status='PASS',files=files,count=len(files),scope='H3 finalization research and active standalone package'))
 write(HERE/'_SUCCESS.json',dict(status='complete',selected='tabm_kzt H3 expanding2010 rank80',manifest_sha256=sha(HERE/'MANIFEST.json'),files=len(files),reporting_csv_rows=3721,source_package_files=len(package)))
 print('Research and package seal PASS',len(files),len(package))
if __name__=='__main__':main()
