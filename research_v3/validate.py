"""Run the required local checks and persist exact commands/exit status."""
from pathlib import Path
import datetime,json,subprocess,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def main():
    commands=[
        [sys.executable,'-m','unittest','discover','-s','final_solution/tests','-v'],
        [sys.executable,'-m','unittest','discover','-s','research_v3/tests','-v'],
        [sys.executable,'final_solution/main.py','--output-dir','final_solution/output-v3-factual'],
        [sys.executable,'final_solution/main.py','--research-v3','--model','basis_train_120m','--policy','selective','--threshold','0.50','--as-of','2025-12-11','--client-context','research_v3/examples/behavior.synthetic.json','--output-dir','research_v3/preview_output/selective'],
        [sys.executable,'final_solution/main.py','--research-v3','--policy','selective','--threshold','1.0','--as-of','2025-12-16','--output-dir','research_v3/preview_output/abstain'],
    ]
    results=[]
    for command in commands:
        completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        result={'command':command,'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr}
        results.append(result)
        print('PASS' if completed.returncode==0 else 'FAIL',' '.join(command),flush=True)
    selective=json.loads((HERE/'preview_output/selective/decision.json').read_text())
    abstain=json.loads((HERE/'preview_output/abstain/decision.json').read_text())
    assertions={'selective_has_candidate':selective['selected'] is not None,
                'selective_is_simulation':selective['behavior']['simulation'] is True,
                'synthetic_never_sends':selective['eligible_to_send'] is False,
                'abstention_is_explicit':abstain['selected'] is None and 'no_market_candidate' in abstain['suppression_reasons']}
    passed=all(r['returncode']==0 for r in results) and all(assertions.values())
    (HERE/'validation.json').write_text(json.dumps({'status':'PASS' if passed else 'FAIL','ran_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':results,'assertions':assertions},ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
