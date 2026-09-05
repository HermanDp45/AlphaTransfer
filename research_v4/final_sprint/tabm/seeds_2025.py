"""Complete the two retrospectively requested 2025 seed controls."""
from pathlib import Path
import sys,time,warnings
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
import pandas as pd
from research_v4.final_sprint.tabm import experiment as x
def main():
    x.initialize()
    x.save(x.HERE/'seeds_2025_protocol.json',dict(created_unix=time.time(),status='Requested after2026 readout to complete matched2025 seed consistency. Retrospective, not untouched validation.',fits=2,configuration='KZT120m seed1/2, same architecture/hyperparameters as existing2026 fits. No tuning. Existing seed0 architecture originally selected on2025.',ensemble='Equal raw probability weights1/3 for seed0/1/2, both2025 and2026; calibrator and policy fitted only on preceding year.',code_sha256=x.sha(__file__)))
    spec=dict(scope='kzt',months=120,seed_index=0)
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,features=x.build()
        for seed in (1,2):x.run(views,features,{**spec,'seed_index':seed},'2025-01-01')
        x.ensemble(spec,'2025-01-01');x.ensemble(spec,'2026-01-01');x.aggregate()
    registry=[]
    for year in (2025,2026):
        components=[]
        for seed in (0,1,2):
            name=x.name({**spec,'seed_index':seed})+f'_{year}-01-01';p=x.CKPT/name
            components.append(dict(model=name,raw_weight=1/3,weights_sha256=x.sha(p/'weights.pt'),preprocessor_sha256=x.sha(p/'preprocess.joblib'),raw_predictions_sha256=x.sha(p/'raw_predictions.joblib')))
        registry.append(dict(config_id=x.name({**spec,'seed_index':'ensemble3'}),cutoff=f'{year}-01-01',components=components,aggregation='mean raw probability, then prior-year Platt and prior-year notification policy',test_fitted_weights=False))
    x.save(x.HERE/'ensemble_registry.json',registry)
    x.save(x.HERE/'seeds_2025_completion.json',dict(status='complete',neural_fits=2,protocol_sha256=x.sha(x.HERE/'seeds_2025_protocol.json'),ensemble_registry_sha256=x.sha(x.HERE/'ensemble_registry.json')))
if __name__=='__main__':main()
