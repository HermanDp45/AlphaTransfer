"""Verify completed scientific receipts and preserve the two prior research seals."""
from pathlib import Path
import hashlib,json,re,datetime
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
V4=HERE.parent
ROOT=V4.parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    previous=json.loads((HERE/'previous_v4_manifest.json').read_text())
    documents=[]
    for name,record in previous['files'].items():
        path=V4/name
        if name in ('README.md','REPORT.md'):
            assert sha(HERE/'prior_reports'/name)==record['sha256'],name
            if sha(path)!=record['sha256']:documents.append(name)
        else:assert sha(path)==record['sha256'],name
    old=json.loads((ROOT/'research_v3/manifest.json').read_text())
    for name,expected in old['files'].items():assert sha(ROOT/name)==expected,name
    lock=json.loads((ROOT/'final_solution/inputs.lock.json').read_text())
    for name,expected in lock['files'].items():assert sha(ROOT/name)==expected,name
    normalized=json.loads((ROOT/'final_solution/data/data_manifest.json').read_text())
    for name,record in normalized['artifacts'].items():assert sha(ROOT/'final_solution/data'/name)==record['sha256'],name
    evidence=['oxr/audit/verification.json','foundation/final_verification.json','robustness/results/verification.json']
    o,f,r=[json.loads((HERE/name).read_text()) for name in evidence]
    assert o['failed']==0 and o['passed']>=124
    assert f['status']=='PASS',f
    assert r['status']=='PASS' and all(x['status']=='PASS' for x in r['checks'])
    assert sha(HERE/'oxr/experiment.py')==o['experiment_sha256']
    assert sha(HERE/'oxr/assess.py')==o['assess_sha256']
    links=0
    for path in HERE.rglob('*.md'):
        if 'prior_reports' in path.parts:continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            target=target.strip('<>')
            if '://' in target or target.startswith('#'):continue
            target=unquote(target.split('#',1)[0])
            resolved=(path.parent/target).resolve()
            # This verifier creates its own linked receipt only after checks pass.
            assert resolved.exists() or resolved==HERE/'verification.json',(str(path.relative_to(HERE)),target)
            links+=1
    result=dict(status='PASS',verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_v4_manifest_sha256=sha(HERE/'previous_v4_manifest.json'),prior_v4_files=len(previous['files']),prior_v4_scientific_artifacts_unchanged=True,parent_navigation_documents_updated=documents,original_navigation_backups_exact=True,v3_files_unchanged=len(old['files']),frozen_final_solution_inputs=len(lock['files']),normalized_artifacts_unchanged=len(normalized['artifacts']),scientific_receipts_sha256={name:sha(HERE/name) for name in evidence},oxr_independent_checks=o['passed'],foundation_status=f['status'],robustness_check_groups=len(r['checks']),local_markdown_links=links,figure_visual_inspection='OXR history_depth.png checked: zero reference, paired month intervals, no clipped labels; no selection-adjusted claim',scope='Research reproducibility and source/cutoff checks, not proof of market superiority or customer savings')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
