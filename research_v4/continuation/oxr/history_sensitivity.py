"""Post-readout sensitivity: isolate source depth in the same basis family.

This does not reselect the primary candidate or create a new holdout.
"""
from pathlib import Path
import time
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import HERE,build_panel,evaluate,save

def main():
    specs=[dict(name=f'oxr_basis_120m_since{year}',months=120,family='basis',delay=24,since=f'{year}-01-01') for year in (2020,2022)]
    save(HERE/'history_sensitivity_protocol.json',dict(timestamp_unix=time.time(),specs=specs,status='exploratory_followup_after_initial_2026_readout',purpose='Hold feature family constant while removing source2018-2019 or2018-2021. No change to primary selection.',test_selection='none'))
    output=[]
    with threadpool_limits(limits=1):
        for spec in specs:
            p=build_panel(24,spec['since'])
            for cutoff in ('2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'):
                output.append(evaluate(p,spec,cutoff,f'{pd.Timestamp(cutoff).year+1}-01-01'))
    pred=pd.concat(output,ignore_index=True)
    pred[pred.fold_test_year.le(2025)].to_csv(HERE/'sensitivity_development_predictions.csv.gz',index=False)
    pred[pred.fold_test_year.eq(2026)].to_csv(HERE/'sensitivity_test_predictions.csv.gz',index=False)

if __name__=='__main__':main()
