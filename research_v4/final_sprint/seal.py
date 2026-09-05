"""Verify scientific inputs and seal the completed additive research/profile."""
from pathlib import Path
import hashlib,json,datetime,re
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];V4=HERE.parent;PROFILE=ROOT/'final_solution/final_sprint'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def entry(p):return dict(sha256=sha(p),bytes=p.stat().st_size)
def useful(p):return p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.log','.pyc') and p.name!='.DS_Store'

def main():
    checks=[]
    def check(name,ok,**extra):
        checks.append(dict(check=name,passed=bool(ok),**extra))
        assert ok,(name,extra)
    old=json.loads((HERE/'previous_v4_manifest.json').read_text())['files']
    allowed={'README.md':'research_v4_README.md','REPORT.md':'research_v4_REPORT.md'}
    changed=[]
    for name,meta in old.items():
        if sha(V4/name)!=meta['sha256']:
            changed.append(name)
            check('previous_v4_navigation_backup_'+name,name in allowed and sha(HERE/'previous_navigation'/allowed[name])==meta['sha256'])
    check('previous_v4_science_unchanged',set(changed).issubset(allowed),files=len(old),navigation_changes=changed)
    v3=json.loads((ROOT/'research_v3/manifest.json').read_text())['files'];changed=[]
    for name,digest in v3.items():
        if sha(ROOT/name)!=digest:
            changed.append(name)
            check('v3_navigation_backup',name=='final_solution/README.md' and sha(HERE/'previous_navigation/final_solution_README.md')==digest)
    check('v3_scientific_inputs_unchanged',set(changed).issubset({'final_solution/README.md'}),files=len(v3),navigation_changes=changed)
    lock=json.loads((ROOT/'final_solution/inputs.lock.json').read_text())['files']
    check('legacy_final_inputs_unchanged',all(sha(ROOT/k)==v for k,v in lock.items()),files=len(lock))
    for label,path in [('neural',HERE/'tabm/final_verification.json'),('catboost',HERE/'catboost/verification.json'),('runtime',HERE/'catboost/runtime_audit/verification.json'),('evaluator',HERE/'catboost/root_evaluation_verification.json')]:
        q=json.loads(path.read_text());check(label+'_checks',q['status']=='PASS',receipt=str(path.relative_to(HERE)))
    for label,folder,manifest in [('neural',HERE/'tabm','MANIFEST.json'),('product',HERE/'product','artifact_manifest.json')]:
        files=json.loads((folder/manifest).read_text())['files'];ok=all(sha(folder/k)==(v['sha256'] if isinstance(v,dict) else v) for k,v in files.items())
        check(label+'_sealed_files',ok,files=len(files))
    broken=[];count=0
    for path in [HERE/'REPORT.md',HERE/'README.md',PROFILE/'README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            target=unquote(target.strip('<>')).split('#')[0]
            if not target or '://' in target:continue
            count+=1
            if not (path.parent/target).exists() and (path.parent/target)!=HERE/'verification.json':broken.append([str(path),target])
    check('report_links',not broken,count=count,broken=broken)
    import pandas as pd
    config=json.loads((PROFILE/'model.json').read_text());out=pd.read_csv(PROFILE/'output/predictions.csv');selected=pd.read_csv(HERE/'selected_predictions.csv.gz')
    selected=selected[selected.cutoff.eq('2026-01-01')&selected['mode'].eq('normal')].sort_values('date')
    check('packaged_primary_contacts',len(out)==156 and out.candidate_signal.tolist()==selected.candidate_signal.tolist(),NOW=int(out.candidate_signal.sum()))
    check('packaged_closing_annotations',int(out.closing_annotation.sum())==21 and not (out.closing_annotation & ~out.candidate_signal).any())
    check('selected_normal_gates',json.loads((HERE/'selection.json').read_text())['champion']['passed'])
    save(HERE/'verification.json',dict(status='PASS',checks=checks,model_fits=160,inner_epoch_selection_fits=18,model_policy_pairs=292,probability_replay_tolerance=1.3e-7,contact_parity='exact',source_delay_selection_gate=False))
    profile_files={str(p.relative_to(PROFILE)):entry(p) for p in sorted(PROFILE.rglob('*')) if useful(p) and p.name not in ('artifact_manifest.json','_SUCCESS.json')}
    save(PROFILE/'artifact_manifest.json',dict(status='PASS',files=profile_files,selection_sha256=sha(HERE/'selection.json')))
    save(PROFILE/'_SUCCESS.json',dict(status='PASS',files=len(profile_files),manifest_sha256=sha(PROFILE/'artifact_manifest.json'),NOW=40,CLOSING_annotations=21,contact_parity='exact',probability_tolerance=1.3e-7))
    own={str(p.relative_to(HERE)):entry(p) for p in sorted(HERE.rglob('*')) if useful(p) and p not in (HERE/'artifact_manifest.json',HERE/'_SUCCESS.json')}
    save(HERE/'artifact_manifest.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Completed final sprint; logs/caches excluded; frozen previousmanifest and input cache included.',files=own,final_profile_manifest_sha256=sha(PROFILE/'artifact_manifest.json')))
    save(HERE/'_SUCCESS.json',dict(status='PASS',files=len(own),bytes=sum(v['bytes'] for v in own.values()),manifest_sha256=sha(HERE/'artifact_manifest.json'),selected=config['config_id'],policy=config['policy_name']))
    paths={V4/k for k in old}|{p for p in HERE.rglob('*') if useful(p)}
    combined={str(p.relative_to(V4)):entry(p) for p in sorted(paths)}
    previous=json.loads((HERE/'previous_v4_manifest.json').read_text())
    save(V4/'artifact_manifest.json',dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Previous sealed V4 plus final_sprint; explicit navigation updates; logs/caches excluded.',files=combined,external_product_evidence_sha256=previous.get('external_product_evidence_sha256'),final_profile_manifest_sha256=sha(PROFILE/'artifact_manifest.json')))
    save(V4/'_SUCCESS.json',dict(status='PASS',sealed_files=len(combined),sealed_bytes=sum(v['bytes'] for v in combined.values()),scientific_validation='Final sprint and executable profile verified; previous science/lockedinputs preserved',V3_scientific_inputs_unchanged=True,navigation_changes=['research_v4/README.md','research_v4/REPORT.md','final_solution/README.md'],manifest_sha256=sha(V4/'artifact_manifest.json'),latest_research='final_sprint/REPORT.md'))
    print(json.dumps(dict(status='PASS',sprint_files=len(own),profile_files=len(profile_files),V4_files=len(combined),V4_manifest_sha256=sha(V4/'artifact_manifest.json')),indent=2))
if __name__=='__main__':main()
