"""Bounded post-hoc combination of long train and Treasury lag7 features."""
from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from benchmark import REPO,BASIS_FEATURES,TREASURY_LAG7_FEATURES,augment_panel,evaluate,score
from experiment import paired_audit
from research_v3.models import experiment as models

def main():
    out=HERE/"long_combo";out.mkdir(exist_ok=True)
    predictions=[];metrics=[];verification=[]
    for historic in (False,True):
        source="long_with_historical_cny" if historic else "basis_train_120m"
        p=models.spec_panel(models.Spec(source,months=120,extended=True,historic_cny=historic))
        p=augment_panel(p)
        for addon in (False,True):
            cid=source+("__treasury_lag7" if addon else "__reproduced")
            print(cid,flush=True)
            feats=BASIS_FEATURES+(TREASURY_LAG7_FEATURES if addon else [])
            pred,cells=evaluate(p,feats,cid,train_years=10,disable_early_stopping=True)
            predictions.append(pred);metrics.append(cells)
            if not addon:
                original=pd.read_csv(REPO/"research_v3/models"/(source+"_h5_predictions.csv.gz"),parse_dates=["date"])
                m=pred.merge(original,on=["date","corridor","fold_test_year"],suffixes=("_new","_original"),validate="one_to_one")
                error=float(abs(m.probability_new-m.probability_original).max())
                assert error<1e-12,(source,error)
                verification.append(dict(source=source,rows=len(m),max_probability_difference=error))
    p=pd.concat(predictions,ignore_index=True);c=pd.concat(metrics,ignore_index=True)
    p.to_csv(out/"predictions.csv.gz",index=False);c.to_csv(out/"fold_corridor_metrics.csv",index=False)
    for label,years in [("development",[2023,2024,2025]),("diagnostic_2026",[2026])]:
        pred=p[p.fold_test_year.isin(years)]
        score(pred,c[c.fold_test_year.isin(years)]).to_csv(out/(label+"_metrics.csv"),index=False)
        pairs=[(s+"__treasury_lag7",s+"__reproduced") for s in ("basis_train_120m","long_with_historical_cny")]
        paired_audit(pred,pairs).to_csv(out/(label+"_paired_audit.csv"),index=False)
    (out/"verification.json").write_text(json.dumps(verification,indent=2))
    (out/"protocol.json").write_text(json.dumps(dict(status="posthoc_combination",train_months=120,validation_months=12,disable_early_stopping=True,features=TREASURY_LAG7_FEATURES,new_macro_feature_coverage_starts="2020-01-01",test_years=[2023,2024,2025],diagnostic_year=2026,production_promotions=0),indent=2))

if __name__=="__main__":main()
