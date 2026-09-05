"""Small, explicitly exploratory combination after individual ablations.

Four combinations on long120m; no outcome-selected blend coefficient.
Incremental comparisons must use long+Treasury or long+MOEX, not just old24m.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research_v4.liquidity.experiment import *
from research_v3.external_data.benchmark import TREASURY_LAG7_FEATURES

def run_combo():
    p,groups,_=build_panel(True,1)
    p=benchmark.augment_panel(p)
    cg=json.loads((ROOT/'research_v4/crypto/output/feature_groups.json').read_text())
    crypto=pd.read_parquet(ROOT/'research_v4/crypto/data/features_daily.parquet')
    names=cg['erub_premium_l2']
    p=p.merge(crypto[['date',*names]],on='date',how='left',validate='many_to_one')
    specs={'treasury_moex':list(TREASURY_LAG7_FEATURES)+groups['moex_liquidity'],'treasury_moex_erub':list(TREASURY_LAG7_FEATURES)+groups['moex_liquidity']+names,'moex_erub':groups['moex_liquidity']+names,'treasury_halyk':list(TREASURY_LAG7_FEATURES)+groups['halyk']}
    results=[];receipts=[]
    with threadpool_limits(limits=1):
        for name,fs in specs.items():
            name='combo_'+name+'_120m';before=core.make_model;models=[]
            def factory(kind,features):
                model=before(kind,features);models.append(model);return model
            core.make_model=factory
            try:pred,cells=benchmark.evaluate(p,old.BASE_FEATURES+[old.BASIS]+fs,name,train_years=10,disable_early_stopping=True)
            finally:core.make_model=before
            pred=pred.merge(p[['date','corridor','rub_per_unit','session_ordinal','pr60']],on=['date','corridor'],validate='one_to_one')
            dest=HERE/'predictions'/f'{name}.csv.gz';pred.to_csv(dest,index=False)
            cells.to_csv(HERE/'predictions'/f'{name}_cells.csv',index=False)
            ckpt=HERE/'checkpoints'/f'{name}.pkl';ckpt.write_bytes(pickle.dumps(models))
            receipts.append({'name':name,'features':old.BASE_FEATURES+[old.BASIS]+fs,'prediction_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'year_order':[2023,2024,2025,2026],'train_years':10,'validation_years':1,'horizon':5,'purge':5,'status':'post-ablation exploratory combination'})
            results.append(pred);print(name,flush=True)
    pd.concat(results,ignore_index=True).to_csv(HERE/'combo_predictions.csv.gz',index=False)
    pd.concat([old.summarize(x) for x in results]).to_csv(HERE/'combo_metrics.csv',index=False)
    (HERE/'combo_receipts.json').write_text(json.dumps(receipts,indent=2))
if __name__=='__main__':run_combo()
