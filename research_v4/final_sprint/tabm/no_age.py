"""Two-fit control for a diagnosed out-of-distribution source-age feature."""
from pathlib import Path
import sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as x
def main():
    x.initialize()
    x.save(x.HERE/'no_age_protocol.json',dict(created_unix=time.time(),fits=2,status='Post-readout numerical attribution control, exploratory.',finding='OXR age nearly constant in normal training (2025 range1.2943..1.3368 days); source-delay mean2.2952 is outside support. Restoring only age to normal fixes2025 delayed Brier0.512112→0.144770 without refit.',change='Exclude only oxr_age_days from the numerical predictive features. Keep oxr_available, all economic OXR basis features, Halyk and Treasury. No winsorization tuned on test, no hyperparameter search.',configuration='KZT120m seed0; unchanged architecture and temporal epoch selection; cutoffs2025/2026; calibration normal prior year; all fixed-model delay views.',code_sha256=x.sha(__file__),base_code_sha256=x.sha(x.__file__),diagnostic_sha256=x.sha(x.HERE/'source_delay_diagnostic.csv')))
    original=x.name;x.name=lambda s:original(s)+'_noage'
    with threadpool_limits(limits=2):
        views,features=x.build();features=[f for f in features if f!='oxr_age_days']
        for cutoff in ('2025-01-01','2026-01-01'):x.run(views,features,dict(scope='kzt',months=120,seed_index=0,feature_control='no_age'),cutoff)
        x.aggregate()
    x.save(x.HERE/'no_age_completion.json',dict(status='complete',neural_fits=2,protocol_sha256=x.sha(x.HERE/'no_age_protocol.json'),code_sha256=x.sha(__file__)))
if __name__=='__main__':main()
