"""Source-specific Kazakhstan continuation, predeclared small family."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research_v4.kazakhstan.experiment import *
from research_v4.liquidity.experiment import build_panel
def main():
    with threadpool_limits(limits=1):
        output=[]
        for lag in (1,2):
            p,groups,_=build_panel(True,lag)
            for source in ('halyk','kase_prices'):
                for strategy in ('pooled','only','residual','residual_shrink'):
                    output.append(evaluate(120,strategy,panel=p,extra_features=groups[source],tag=f'__{source}_lag{lag}'))
        pd.concat(output,ignore_index=True).to_csv(HERE/'context_predictions.csv.gz',index=False)
        pd.concat([old.summarize(p) for p in output],ignore_index=True).to_csv(HERE/'context_metrics.csv',index=False)
if __name__=='__main__':main()
