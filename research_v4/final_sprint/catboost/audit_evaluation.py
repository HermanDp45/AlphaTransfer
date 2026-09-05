"""Read-only independent final-sprint leaderboard, policy registry and CI audit."""
from pathlib import Path
import sys,json,hashlib,gzip,io
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import experiment as x
import verify as verifier
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.final_sprint import common

def independent_metrics(q):
    chosen=q.candidate_signal.to_numpy(bool);target=q.target.to_numpy(float);forward=q.forward_bps.to_numpy(float)
    week0=q.date.min().to_period('W-SUN').ordinal;week1=q.date.max().to_period('W-SUN').ordinal
    indices=np.array([d.to_period('W-SUN').ordinal-week0 for d in q.date])
    counts=np.bincount(indices[chosen],minlength=week1-week0+1)
    return dict(rows=len(q),dates=q.date.nunique(),signals=int(chosen.sum()),weeks=len(counts),
        weeks_1_2=float(np.mean((counts>=1)&(counts<=2))),signals_per_week=float(counts.mean()),silent_weeks=int((counts==0).sum()),
        brier=float(np.mean((q.probability-target)**2)),base_hit=float(target.mean()),hit_rate=float(target[chosen].mean()),
        lift=float(target[chosen].mean()/target.mean()),forward_delta_bps=float(forward[chosen].mean()-forward.mean()),
        forward_signal_bps=float(forward[chosen].mean()))

def main():
    root=HERE.parent;checks=[]
    files=['policy_predictions.csv.gz','leaderboard.csv','ranking.csv','selection.json','external_policy_registry.json',
           'selected_predictions.csv.gz','selected_intervals.json','protocol.json','evaluate.py','common.py']
    data={name:(root/name).read_bytes() for name in files}
    hashes={name:hashlib.sha256(body).hexdigest() for name,body in data.items()}
    def read(name):
        body=gzip.decompress(data[name]) if name.endswith('.gz') else data[name]
        return pd.read_csv(io.BytesIO(body),parse_dates=['date'] if 'predictions' in name else None)
    def check(name,passed,**details):
        checks.append(dict(check=name,passed=bool(passed),**details));print(name,'PASS' if passed else 'FAIL',flush=True)
    pred=read('policy_predictions.csv.gz');table=read('leaderboard.csv');ranking=read('ranking.csv')
    selection=json.loads(data['selection.json']);registry=json.loads(data['external_policy_registry.json'])
    keys=['config_id','cutoff','mode','policy'];errors=[];maximum=0.
    for key,q in pred.groupby(keys):
        calculated=independent_metrics(q);row=table
        for column,value in zip(keys,key):row=row[row[column].eq(value)]
        if len(row)!=1:errors.append(str(key));continue
        for column,value in calculated.items():
            delta=abs(value-float(row.iloc[0][column]));maximum=max(maximum,delta)
            if delta>1e-11:errors.append(str(key)+column)
    check('leaderboard_all_rows_independent_metrics',not errors,metric_rows=len(table),maximum_numeric_error=maximum,errors=errors[:5])
    cal=pd.read_csv(HERE/'candidates_calibration_predictions.csv.gz',parse_dates=['date'])
    hist=pd.read_csv(HERE/'candidates_history_predictions.csv.gz',parse_dates=['date'])
    errors=[];count=0
    for entry in registry:
        name=entry['config_id'];cutoff=entry['cutoff']
        if not name.startswith('catboost_'):continue
        count+=1
        c=cal[cal.config_id.eq(name)&cal.cutoff.eq(cutoff)].reset_index(drop=True)
        h=hist[hist.config_id.eq(name)&hist.cutoff.eq(cutoff)].reset_index(drop=True)
        regenerated=common.fit_policy(c,h,entry['name'])
        actual={key:entry[key] for key in ('name','threshold','cooldown','initial_state')}
        if regenerated!=actual:errors.append(name+cutoff+'fit')
        for mode,q in pred[pred.config_id.eq(name)&pred.cutoff.eq(cutoff)&pred.policy.eq(entry['name'])].groupby('mode'):
            ids,_=verifier.direct_select(q,actual,actual['initial_state'])
            if not np.array_equal(q.index.isin(ids),q.candidate_signal):errors.append(name+cutoff+mode+'mask')
    check('all_external_CatBoost_registry_and_actual_policy_masks',count==72 and not errors,policy_records=count,errors=errors[:5])
    champion=selection['champion'];check('selection_discloses_retrospective_2026',selection['selection_uses_2026'] is True)
    errors=[]
    for row in ranking.itertuples():
        family=table[table.config_id.eq(row.config_id)&table.policy.eq(row.policy)]
        expected={'normal'} if row.config_id=='v3_120m' else {'normal','bank_delayed'}
        if 'oxr' in row.config_id or 'tabm' in row.config_id:expected|={'oxr_delayed','both_delayed'}
        passes=[];fresh=bool(selection.get('fresh_bank_override',False))
        for year in (2025,2026):
            q=family[family.cutoff.eq(f'{year}-01-01')]
            if (year==2026 or not fresh) and not expected.issubset(set(q['mode'])):errors.append(row.config_id+'missingstress'+str(year))
            if fresh:q=q[q['mode'].eq('normal')]
            passes.append(bool((not fresh or len(q)==1) and q.lift.min()>=1.3-1e-12 and q.weeks_1_2.min()>=.85 and q.forward_delta_bps.min()>0))
        if bool(row.passed)!=all(passes):errors.append(row.config_id+'wrongeligibility')
    check('registry_expected_diagnostics_and_user_authorized_2025_2026_gates',not errors,model_policy_rows=len(ranking),fresh_bank_override=fresh,errors=errors[:5])
    if fresh:
        expected_order=ranking.sort_values(['passed','normal_lift','normal_utility_bps','normal_2025_lift','redundancy','policy_complexity'],ascending=[False,False,False,False,True,True])
        check('normal_primary_champion_ranking_order',expected_order.iloc[0].config_id==champion['config_id'] and expected_order.iloc[0].policy==champion['policy'])
    chosen=ranking[ranking.config_id.eq(champion['config_id'])&ranking.policy.eq(champion['policy'])].iloc[0]
    check('champion_registry_identity',all(chosen[k]==champion[k] if isinstance(champion[k],str) else np.isclose(chosen[k],champion[k]) for k in ('config_id','policy','min_lift','min_coverage','min_utility_bps')))
    selected=read('selected_predictions.csv.gz');stored=json.loads(data['selected_intervals.json']);errors=[];maximum=0.
    for (year,mode),q in selected.groupby(['fold_test_year','mode']):
        # Independent four-calendar-week blocks, explicitly summed from raw date rows.
        begin=q.date.min().to_period('W-SUN').ordinal
        blocks=np.array([(d.to_period('W-SUN').ordinal-begin)//4 for d in q.date])
        target=q.target.to_numpy(float);forward=q.forward_bps.to_numpy(float);signals=q.candidate_signal.to_numpy(bool)
        components=[]
        for b in sorted(set(blocks)):
            mask=blocks==b;marked=mask&signals
            components.append([mask.sum(),target[mask].sum(),marked.sum(),target[marked].sum(),forward[mask].sum(),forward[marked].sum()])
        components=np.array(components);rng=np.random.default_rng(20260905)
        index=rng.integers(len(components),size=(10000,len(components)))
        summed=components[index].sum(axis=1)
        lifts=(summed[:,3]/summed[:,2])/(summed[:,1]/summed[:,0]);utility=summed[:,5]/summed[:,2]-summed[:,4]/summed[:,0]
        row=next(r for r in stored if r['year']==year and r['mode']==mode and r['kind']=='absolute_four_week')
        error=max(np.max(abs(np.quantile(lifts,[.025,.975])-row['lift_ci95'])),
                  np.max(abs(np.quantile(utility,[.025,.975])-row['forward_delta_bps_ci95'])))
        maximum=max(maximum,float(error))
        if error>1e-10:errors.append(str(year)+mode)
    check('all_selected_absolute_four_week_intervals_independent',not errors,maximum_numeric_error=maximum,errors=errors)
    after={name:x.sha(root/name) for name in files}
    check('root_inputs_not_changed_during_audit',after==hashes)
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',passed=sum(c['passed'] for c in checks),
        failed=sum(not c['passed'] for c in checks),checks=checks,root_input_hashes=hashes,champion_at_audit=champion,
        tree_fits=0,limitations='Conditional fixed-prediction intervals; does not repeat retrospective selection over the188-or-later family. Positive relative reference-rate advantage is not executable net profit.',
        verifier_sha256=x.sha(__file__))
    x.save(HERE/'root_evaluation_verification.json',result)
    print(json.dumps({k:result[k] for k in ('status','passed','failed')}),flush=True)
    if result['failed']:raise SystemExit(1)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
