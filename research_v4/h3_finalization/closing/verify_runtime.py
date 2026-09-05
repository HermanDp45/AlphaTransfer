"""Run with python -I: verify standalone sklearn loading without project imports."""
from pathlib import Path
import sys,json,joblib,pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    assert not any('AlphaTransfer' in path for path in sys.path)
    model=joblib.load(HERE/'final/model.joblib');receipt=json.loads((HERE/'final/receipt.json').read_text())
    panel=pd.read_pickle(ROOT/'research_v4/h3_finalization/audit/latest_panel.pkl')
    q=pd.read_csv(HERE/'final/raw_predictions.csv.gz',parse_dates=['date']);q=q[q.split.eq('tail')]
    x=panel.set_index('date').loc[q.date].reset_index();prob=model.predict_proba(x[receipt['features']+['corridor']])[:,1]
    np.testing.assert_allclose(prob,q.raw_probability.to_numpy(),atol=1e-15,rtol=0)
    result=dict(status='PASS',isolated_python=True,project_import_paths_absent=True,loaded_type=type(model).__module__+'.'+type(model).__name__,tail_rows=len(q),maximum_error=float(np.max(np.abs(prob-q.raw_probability.to_numpy()))))
    (HERE/'isolated_runtime_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':main()
