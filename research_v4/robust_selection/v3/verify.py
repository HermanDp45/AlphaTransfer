"""Independent checkpoint, temporal maturity, outcome and scheduler audits."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.v3 import experiment as e
HERE=e.HERE

def independent_select(history,test):
    frame=pd.concat([history,test]).sort_values(['date','corridor']).reset_index(drop=True)
    candidate=np.zeros(len(frame),bool);last={}
    for i,r in frame.iterrows():
        if r.probability>=.5 and r.session_ordinal-last.get(r.corridor,-10000)>3:
            candidate[i]=True;last[r.corridor]=int(r.session_ordinal)
    portfolio=np.zeros(len(frame),bool);last_session=-10000
    for day,g in frame[candidate].groupby('date',sort=True):
        high=g[g.probability.eq(g.probability.max())]
        best=min(high.index,key=lambda i:(int.from_bytes(hashlib.sha256(f'{e.core.SEED}|{day.date().isoformat()}|{frame.loc[i,"corridor"]}'.encode()).digest()[:8],'big'),frame.loc[i,'corridor'],i))
        if frame.loc[best,'session_ordinal']-last_session>3:portfolio[best]=True;last_session=int(frame.loc[best,'session_ordinal'])
    frame['candidate']=candidate;frame['portfolio']=portfolio
    return frame[frame.split.eq('test')].sort_values(['date','corridor'])

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());assert e.sha(HERE/'experiment.py')==protocol['code_sha256']
    for f,h in protocol['inputs'].items():assert e.sha(ROOT/f)==h
    pred=pd.read_csv(HERE/'raw_predictions.csv.gz',parse_dates=['date','label_available_date']);warm=pd.read_csv(HERE/'warmup.csv.gz',parse_dates=['date','label_available_date'])
    raw=pd.read_pickle(e.SOURCE);checks=[{'check':'immutable_input_hashes','status':'PASS','files':len(protocol['inputs'])}]
    with threadpool_limits(limits=1):
      for receipt in json.loads((HERE/'model_receipts.json').read_text()):
        h=receipt['train_horizon'];year=receipt['year'];p=e.target_panel(raw,h);parts=e.parts(p,h,year);cut=pd.Timestamp(year,1,1);calstart=pd.Timestamp(year-1,1,1);end=pd.Timestamp(year+1,1,1)
        actual_end=p.groupby('corridor').date.shift(-h)
        for part,boundary in [('train',calstart),('validation',cut),('test',end)]:
            q=parts[part];assert actual_end.loc[q.index].lt(boundary).all();pd.testing.assert_series_equal(actual_end.loc[q.index],q.label_available_date,check_names=False)
        # Independent direct horizon arrays include all eligible raw observations.
        max_utility_error=0.
        for _,g in p.groupby('corridor'):
            rates=g.rub_per_unit.to_numpy();values=g[e.OUTCOME].to_numpy();truth=[]
            for i in range(len(g)-h):
                future=rates[i+1:i+h+1];anchor=rates[i]
                target=float((min(future)/anchor-1)*10000+1e-12>=0)
                forward=(sum(future)/h/anchor-1)*10000
                regret=(anchor/min(anchor,min(future))-1)*10000
                symmetric=(sum(rates[i-h:i+h+1])/(2*h+1)/anchor-1)*10000 if i>=h else np.nan
                expected=np.array([target,forward,symmetric,regret]);np.testing.assert_allclose(values[i],expected,rtol=0,atol=1e-8,equal_nan=True)
                max_utility_error=max(max_utility_error,float(np.nanmax(np.abs(values[i]-expected))))
        cp=HERE/receipt['checkpoint'];assert e.sha(cp)==receipt['checkpoint_sha256'];b=joblib.load(cp)
        g=pred[pred.train_horizon.eq(h)&pred.fold_test_year.eq(year)]
        max_raw=0.;max_cal=0.
        for split_name in ['validation','history','test']:
            q=parts[split_name];saved=g[g.split.eq(split_name)].sort_values(['date','corridor']).reset_index(drop=True);q=q.sort_values(['date','corridor'])
            pd.testing.assert_frame_equal(q[['date','corridor']].reset_index(drop=True),saved[['date','corridor']],check_dtype=False)
            fresh=b['model'].predict_proba(q[e.FEATURES+['corridor']])[:,1];max_raw=max(max_raw,float(np.max(abs(fresh-saved.raw_probability))))
            cal=b['calibrator'];logit=np.log(np.clip(fresh,1e-6,1-1e-6)/(1-np.clip(fresh,1e-6,1-1e-6)))
            manual=fresh if cal.model is None else 1/(1+np.exp(-(cal.intercept+cal.slope*logit)))
            max_cal=max(max_cal,float(np.max(abs(manual-saved.probability))))
            if split_name=='history':assert saved.loc[saved.label_available_date.ge(cut)|saved.label_available_date.isna(),e.OUTCOME].isna().all().all()
        assert max(max_raw,max_cal)<1e-12
        # Future values, labels crossing either boundary cannot affect trained weights/calibration.
        poison=p.copy();poison.loc[poison.date.ge(calstart),e.FEATURES]=poison.loc[poison.date.ge(calstart),e.FEATURES]*100
        poison.loc[poison.date.lt(calstart)&poison.label_available_date.ge(calstart),'target']=1-poison.loc[poison.date.lt(calstart)&poison.label_available_date.ge(calstart),'target']
        poisoned_train=e.parts(poison,h,year)['train'];pd.testing.assert_frame_equal(poisoned_train[e.FEATURES+['target']],parts['train'][e.FEATURES+['target']],check_exact=True)
        test=parts['test'];again=e.old.make_model(e.SPEC,poisoned_train).fit(poisoned_train[e.FEATURES+['corridor']],poisoned_train.target.astype(int))
        np.testing.assert_array_equal(again.predict_proba(test[e.FEATURES+['corridor']]),b['model'].predict_proba(test[e.FEATURES+['corridor']]))
        # Same cutoff history, then test, independently replay the strict original probability gate.
        his=g[g.split.eq('history')];te=g[g.split.eq('test')].sort_values(['date','corridor']);replayed=independent_select(his,te)
        np.testing.assert_array_equal(replayed.candidate.to_numpy(),te.strict05_candidate_signal.to_numpy());np.testing.assert_array_equal(replayed.portfolio.to_numpy(),te.strict05_signal.to_numpy())
        poisoned_g=g.copy();poisoned_g[e.OUTCOME]=999999.;again_mask=independent_select(poisoned_g[poisoned_g.split.eq('history')],poisoned_g[poisoned_g.split.eq('test')]);np.testing.assert_array_equal(again_mask[['candidate','portfolio']],replayed[['candidate','portfolio']])
        w=warm[warm.train_horizon.eq(h)&warm.fold_test_year.eq(year)];assert w.date.nunique()==63 and w.date.max()==p[p.date.lt(calstart)].date.max() and w[e.OUTCOME].isna().all().all();assert (~w.in_sample_training_warmup).sum()==h*5;wp=p.set_index(['date','corridor']).loc[pd.MultiIndex.from_frame(w[['date','corridor']])].reset_index()
        np.testing.assert_allclose(b['model'].predict_proba(wp[e.FEATURES+['corridor']])[:,1],w.raw_probability,rtol=0,atol=1e-12)
        checks.append({'check':'fold_independent_audit','status':'PASS','train_horizon':h,'year':year,'max_raw_checkpoint_error':max_raw,'max_manual_platt_error':max_cal,'max_direct_utility_error':max_utility_error,'actual_label_maturity':True,'future_features_and_unmatured_train_labels_refit_invariant':True,'strict05_independent_candidate_and_portfolio_replay':True,'outcome_poison_scheduler_invariant':True,'warmup_63_panel_dates_including_purged_tail_same_checkpoint_all_outcomes_masked':True})
    assert json.loads((HERE/'historical_parity.json').read_text())['status']=='PASS'
    e.save(HERE/'verification.json',{'status':'PASS','checks':checks,'historical_h3_and_h5_parity':'PASS','fits_recomputed_for_future_invariance':6})
    print('PASS',len(checks),'checks')
if __name__=='__main__':main()
