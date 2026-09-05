"""Seal/verify the completed V4 artifact tree, excluding only mutable caches.

No model training, download, message delivery or writes to sealed V3.
"""
from pathlib import Path
import argparse,hashlib,json,datetime,re
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'artifact_manifest.json'
EXCLUDED={'artifact_manifest.json','_SUCCESS.json','verify_package.log'}

def files():
    return sorted(p for p in HERE.rglob('*') if p.is_file() and not any(x in ('__pycache__','.DS_Store') for x in p.parts) and p.suffix not in ('.pyc','.log') and str(p.relative_to(HERE)) not in EXCLUDED)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def links():
    checked=0
    for filename in ('REPORT.md','README.md','kazakhstan/REPORT.md','liquidity/README.md'):
        file=HERE/filename
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#',1)[0]
            assert (file.parent/target).exists(),(filename,target)
            checked+=1
    return checked

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    # Scientific verification output is an input to the package seal, not a
    # replacement for rerunning verify_market/foundation/crypto/unit tests.
    scientific=json.loads((HERE/'market_verification.json').read_text());assert scientific['status']=='PASS'
    assert json.loads((HERE/'foundation/verification_receipt.json').read_text())['status']=='passed'
    assert all(x['passed'] for x in json.loads((HERE/'crypto/output/verification.json').read_text()))
    assert json.loads((HERE/'validation.json').read_text())['status']=='PASS'
    external=HERE.parent/'product_artifacts/V4_SEGMENTATION_EVIDENCE.md'
    checked_links=links()
    if args.seal:
        manifest={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'V4 research artifacts; logs/caches excluded, raw WB microdata external and not redistributed','external_product_evidence_sha256':digest(external),'files':{str(p.relative_to(HERE)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in files()}}
        MANIFEST.write_text(json.dumps(manifest,indent=2))
    manifest=json.loads(MANIFEST.read_text())
    assert digest(external)==manifest['external_product_evidence_sha256']
    actual={str(p.relative_to(HERE)) for p in files()}
    assert actual==set(manifest['files']),'Unsealed or missing artifact files'
    for name,meta in manifest['files'].items():
        path=HERE/name
        assert path.stat().st_size==meta['bytes'] and digest(path)==meta['sha256'],name
    result={'status':'PASS','sealed_files':len(actual),'sealed_bytes':sum(x['bytes'] for x in manifest['files'].values()),'local_report_links':checked_links,'scientific_validation':'market verification PASS; separate foundation/crypto checks and segment tests retained','V3_untouched':scientific['status']=='PASS','manifest_sha256':digest(MANIFEST)}
    (HERE/'_SUCCESS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
