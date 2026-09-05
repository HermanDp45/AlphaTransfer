"""Verify exact frozen setup before allowing reuse of saved neural checkpoints."""
from pathlib import Path
import sys,json,pickle
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd
from research_v4.architecture_2023_2025.pooled import experiment as e
def audit():
    protocol=json.loads((e.HERE/'protocol.json').read_text())
    assert e.sha(e.__file__)==protocol['code_sha256'],'Experiment code differs from frozen protocol'
    for name,digest in protocol['inputs'].items():assert e.sha(ROOT/name)==digest,f'Frozen input differs: {name}'
    views,_=pickle.loads(e.SOURCE.read_bytes());panel=views['2010-01-01',24,1]
    rows=[]
    for year in (2023,2024,2025):
        parts=e.split(panel,year);tr=parts['train']
        for fs,features in e.FEATURE_SETS.items():
            stem=f'{fs}_{year}';nn=e.CHECKPOINTS/f'tabm_{stem}';tree=e.CHECKPOINTS/f'hgb_{stem}'
            if not (nn/'model.json').exists():continue
            assert (nn/'split_receipt.json').exists(),'Incomplete prior fit: inspect rather than silently reuse'
            meta=json.loads((nn/'model.json').read_text());receipt=json.loads((nn/'split_receipt.json').read_text())
            assert meta['seed']==e.SEED and meta['refit_seed']==e.SEED+1
            assert meta['features']==features and meta['train_rows']==len(tr)
            assert receipt['training_feature_sha256']==e.fingerprint(tr[['date','corridor',*features]])
            assert receipt['source_sha256']==e.sha(e.SOURCE)
            assert receipt['code_sha256']==protocol['code_sha256']
            assert meta['weights_sha256']==e.sha(nn/'weights.pt')
            assert receipt['shared_preprocessor_sha256']==e.sha(nn/'preprocess.joblib')
            assert receipt['hgb_checkpoint_sha256']==e.sha(tree/'model.joblib')
            rows.append(dict(year=year,feature_set=fs,checkpoint_and_setup='PASS',rows=len(tr)))
    return rows
if __name__=='__main__':
    rows=audit();print('Frozen setup verified:',len(rows),'completed pairs',flush=True)
    if '--check-only' not in sys.argv:e.main()
