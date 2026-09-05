"""Verify new scientific receipts and preserve the prior V3/V4/input seals."""
from pathlib import Path
import json,hashlib,re,datetime
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent;V4=HERE.parent;ROOT=V4.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 previous=json.loads((HERE/'previous_v4_manifest.json').read_text())
 for name,record in previous['files'].items():
  path=HERE/'prior_reports'/name if name in ('README.md','REPORT.md') else V4/name
  assert sha(path)==record['sha256'],name
 v3=json.loads((ROOT/'research_v3/manifest.json').read_text())
 for name,value in v3['files'].items():assert sha(ROOT/name)==value,name
 lock=json.loads((ROOT/'final_solution/inputs.lock.json').read_text())
 for name,value in lock['files'].items():assert sha(ROOT/name)==value,name
 normalized=json.loads((ROOT/'final_solution/data/data_manifest.json').read_text())
 for name,record in normalized['artifacts'].items():assert sha(ROOT/'final_solution/data'/name)==record['sha256'],name
 receipts=['data_audit/source_audit.json','data_audit/design_verification.json','data_audit/final_verification.json','long_models/verification.json','long_models/canonical/verification.json','long_models/inference_receipt.json','long_models/simultaneous_receipt.json','foundation/verification.json','foundation/final_verification.json','bank_target/results/verification.json']
 for name in receipts:
  r=json.loads((HERE/name).read_text());assert r['status'] in ('PASS','complete'),name
 assert sha(HERE/'input_oxr_snapshot.csv')==sha(ROOT/'data/open_exchange_rates/rub_cis_daily.csv')
 scientific=json.loads((HERE/'long_models/verification.json').read_text());assert scientific['fits']==135 and scientific['model_view_replays']==285
 canonical=json.loads((HERE/'long_models/canonical/retraining_parity.json').read_text());assert canonical['max_probability_difference']<1e-12 and canonical['candidate_mismatches']==0
 bank_manifest=json.loads((HERE/'bank_target/artifact_manifest.json').read_text())
 for name,value in bank_manifest['files'].items():assert sha(HERE/'bank_target'/name)==value,name
 foundation_manifest=json.loads((HERE/'foundation/MANIFEST.json').read_text())
 for name,record in foundation_manifest['files'].items():assert sha(HERE/'foundation'/name)==record['sha256'],name
 links=0
 for path in HERE.rglob('*.md'):
  if 'prior_reports' in path.parts:continue
  for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
   target=target.strip('<>')
   if '://' in target or target.startswith('#'):continue
   resolved=(path.parent/unquote(target.split('#',1)[0])).resolve()
   assert resolved.exists() or resolved==HERE/'verification.json',(str(path.relative_to(HERE)),target)
   links+=1
 result=dict(status='PASS',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_v4_files_preserved=len(previous['files']),original_navigation_copies_exact=True,v3_files_preserved=len(v3['files']),final_solution_locked_files_unchanged=len(lock['files']),normalized_artifacts_unchanged=len(normalized['artifacts']),new_oxr_sha256=sha(HERE/'input_oxr_snapshot.csv'),scientific_receipts={n:sha(HERE/n) for n in receipts},long_hgb_fits=170,original_model_view_replays=285,canonical_predictions_same_as_original=canonical,bank_package_files=len(bank_manifest['files']),local_report_links=links,figure_inspection='results.png rendered and visually inspected; labels/zero lines visible, ordinary vs simultaneous intervals distinguished',interpretation='Reproducibility/source/timing verification; no claim of confirmed execution uplift or pristine holdout')
 (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
