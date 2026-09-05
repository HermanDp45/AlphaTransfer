from __future__ import annotations
import copy,hashlib,json,sys,unittest
from datetime import date,timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from final_solution.tabm_h3 import features as f,policy


def config():
    return {'schema_version':1,'profile_id':'test','model_id':'test_h3','model_cutoff':'2026-01-01','train_horizon':3,'corridor':'KZT','features':f.FEATURES,'model':{'weights_sha256':'a','preprocessor_sha256':'b'},'calibration':{'method':'identity'},'policy':{'name':'rank80','kind':'rank','threshold':.5,'window':63,'min_history':20,'cooldown_sessions':2,'max_contacts_per_week':2}}


def scores(n=100,start=date(2025,1,1),offset=0):
    return [{'date':str(start+timedelta(days=i)),'corridor':'KZT','session_ordinal':offset+i,'probability':float((i%11)/10)} for i in range(n)]


def sources():
    days=pd.bdate_range('2023-01-02',periods=340);x=np.arange(len(days),dtype=float);values=.17*np.exp(.001*np.sin(x/7)+.0001*x)
    cbr=pd.DataFrame({'date':days,'corridor':'KZT','rub_per_unit':values})
    close=pd.DataFrame({'TRADEDATE':days,'CLOSE':9+np.sin(x/20)/10})
    fixing=pd.DataFrame({'TRADEDATE':days,'CLOSE':9+np.sin(x/20)/10-.01})
    oxr=pd.DataFrame({'date':days,'quote':'KZT','rub_per_quote':values*1.001,'published_at_utc':days.tz_localize('UTC')+pd.Timedelta(hours=17)})
    bank=pd.concat([pd.DataFrame({'date':days,'client':cl,'currency':cc,'bank_side':'sell','value':mult*(5+np.sin(x/20)/10)}) for cl,cc,mult in [('personal','RUB',1.),('legal','RUB',.99),('personal','USD',95.)]],ignore_index=True)
    t1=pd.DataFrame({'source_date':days,'treasury_t10yie':2+np.sin(x/11)/10});t5=pd.DataFrame({'source_date':days,'treasury_t5yifr':2.1+np.sin(x/12)/10})
    return dict(cbr=cbr,moex_cny_close=close,moex_cny_fixing=fixing,oxr=oxr,halyk=bank,treasury_t10yie=t1,treasury_t5yifr=t5)


class PolicyTests(unittest.TestCase):
    def test_full_batch_equals_arbitrary_incremental_batches(self):
        c=config();rows=scores();first,_=policy.replay(rows,c);st=policy.empty_state(c);got=[]
        for l,r in [(0,21),(21,22),(22,63),(63,84),(84,100)]:
            part,st=policy.replay(rows[l:r],c,st);got.extend(part)
        self.assertEqual(first,got)
        self.assertEqual(len(st['past_scores']),63)

    def test_current_and_future_scores_cannot_enter_past_rank(self):
        c=config();r=scores(40);a,_=policy.replay(r,c);r2=copy.deepcopy(r);r2[31]['probability']=1-r2[31]['probability'];b,_=policy.replay(r2,c)
        self.assertEqual(a[:31],b[:31]);expected=(sum(x['probability']<r[30]['probability'] for x in r[:30])+.5*sum(x['probability']==r[30]['probability'] for x in r[:30]))/30
        self.assertEqual(a[30]['rank_score'],expected)

    def test_equal_scores_midranks_and_weekly_cap(self):
        c=config();c['policy']['threshold']=.4;r=scores(42)
        for x in r:x['probability']=.6
        out,_=policy.replay(r,c);self.assertTrue(all(x['rank_score']==.5 for x in out[20:]));buckets={};last=None
        for x in out:
            if x['candidate_signal']:
                week=date.fromisoformat(x['date']).isocalendar()[:2];buckets[week]=buckets.get(week,0)+1
                if last is not None:self.assertGreater(x['session_ordinal']-last,2)
                last=x['session_ordinal']
        self.assertLessEqual(max(buckets.values()),2)

    def test_binding_rejects_changed_calibrator_policy_or_horizon(self):
        c=config();_,s=policy.replay(scores(25),c)
        for key,val in [('train_horizon',5),('calibration',{'method':'identity','revision':1}),('policy',{**c['policy'],'threshold':.6})]:
            different=copy.deepcopy(c);different[key]=val
            with self.assertRaises(ValueError):policy.replay(scores(1,date(2025,1,26),25),different,s)

    def test_state_is_immutable_future_state_and_skipped_sessions_rejected(self):
        c=config();_,s=policy.replay(scores(25),c);before=copy.deepcopy(s)
        policy.replay(scores(1,date(2025,1,26),25),c,s);self.assertEqual(s,before)
        for offset in (24,26):
            with self.assertRaises(ValueError):policy.replay(scores(1,date(2025,1,26),offset),c,s)
        bad=copy.deepcopy(s);bad['last_candidate_session']=30
        with self.assertRaises(ValueError):policy.validate_state(bad,c)

    def test_warmup_scores_do_not_create_notifications_and_outcomes_rejected(self):
        c=config();r=scores(63);out,s=policy.replay(r,c,emit_candidates=False)
        self.assertFalse(any(x['candidate_signal'] for x in out));self.assertEqual(len(s['past_scores']),63);self.assertIsNone(s['last_candidate_session'])
        r=scores(1);r[0]['target']=1
        with self.assertRaises(ValueError):policy.replay(r,c)


class FeatureTests(unittest.TestCase):
    def test_future_poison_and_prefix_invariance(self):
        raw=sources();cut=raw['cbr'].date.iloc[309];full=f.build_features(raw,cut)
        poisoned=copy.deepcopy(raw)
        for name,z in poisoned.items():
            dc='TRADEDATE' if name.startswith('moex') else 'source_date' if name.startswith('treasury') else 'date';future=pd.to_datetime(z[dc]).gt(cut)
            for col in z.select_dtypes(include='number'):z.loc[future,col]*=100
        again=f.build_features(poisoned,cut);pd.testing.assert_frame_equal(full,again,check_exact=True)
        longer=f.build_features(raw,raw['cbr'].date.max());pd.testing.assert_frame_equal(full[f.FEATURES],longer[longer.date.le(cut)][f.FEATURES].reset_index(drop=True),check_exact=True)
        self.assertFalse({'target','future_min','forward_bps','regret_bps'}&set(full))

    def test_published_after_decision_is_not_available_early(self):
        raw=sources();decision=raw['cbr'].date.iloc[-1];before=f.build_features(raw,decision)
        raw['oxr'].loc[raw['oxr'].date.gt(decision-pd.Timedelta(days=12)),'published_at_utc']=decision.tz_localize('UTC')+pd.Timedelta(days=10)
        delayed=f.build_features(raw,decision);self.assertEqual(delayed.oxr_available.iloc[-1],0.);self.assertTrue(pd.isna(delayed.oxr_log_basis.iloc[-1]));self.assertEqual(before.oxr_available.iloc[-1],1.)

    def test_bank_side_orientation_and_same_session_gap(self):
        raw=sources();q=f.build_features(raw,raw['cbr'].date.max());last=q.iloc[-1]
        self.assertAlmostEqual(last.halyk_rub_personal_legal_gap,np.log(1/.99));self.assertGreater(last.halyk_rub_sell_official_basis,-1.)
        raw['halyk']['bank_side']='buy'
        with self.assertRaises(ValueError):f.build_features(raw,raw['cbr'].date.max())

    def test_intraday_and_duplicate_rows_rejected(self):
        raw=sources()
        with self.assertRaises(ValueError):f.build_features(raw,'2024-01-01T01:00:00+03:00')
        raw['cbr']=pd.concat([raw['cbr'],raw['cbr'].iloc[[-1]]])
        with self.assertRaises(ValueError):f.build_features(raw,'2026-09-05')

    def test_complete_history_required_by_session_state(self):
        raw=sources();q=f.build_features(raw,raw['cbr'].date.max());self.assertTrue(q.session_ordinal.eq(range(len(q))).all());self.assertEqual(len(f.FEATURES),33)
        self.assertTrue(q.ret60.iloc[:60].isna().all());self.assertTrue(q.pr252.iloc[:251].isna().all())



class ClosingTests(unittest.TestCase):
    def cfg(self,enabled=False):
        c=config();c['closing']={'enabled':enabled,'scenario':'CLOSING','train_horizon':3,'target':'R[t+3]>R[t]','features':f.FEATURES,'model':'closing.joblib','model_sha256':hashlib.sha256(b'fixture').hexdigest(),'calibration':{'method':'identity'},'threshold':.5,'requires_now':True,'requires_positive_ret1':True,'extra_contacts':False,'status':'diagnostic_only_failed_annual_annotation_gates' if not enabled else 'validated'};return c

    def frame(self):
        x=pd.DataFrame({name:[0.,0.] for name in f.FEATURES});x['corridor']='KZT';x['ret1']=[.01,.01];return x

    def test_disabled_diagnostic_head_cannot_change_primary_scenario(self):
        import tempfile
        from unittest.mock import patch,Mock
        from final_solution.tabm_h3.closing import score_annotation
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'closing.joblib').write_bytes(b'fixture');model=Mock();model.predict_proba.return_value=np.array([[.2,.8],[.1,.9]])
            with patch('final_solution.tabm_h3.closing.joblib.load',return_value=model):
                meta,p,condition,active=score_annotation(self.frame(),[True,False],self.cfg(),temp)
            np.testing.assert_array_equal(condition,[True,False]);self.assertFalse(active.any());self.assertEqual(meta['train_horizon'],3);self.assertEqual(p[0],.8)

    def test_h5_alias_and_failed_enabled_policy_rejected(self):
        from final_solution.tabm_h3.closing import score_annotation
        c=self.cfg();c['closing']['train_horizon']=5
        with self.assertRaises(ValueError):score_annotation(self.frame(),[True,True],c,'.')
        c=self.cfg();c['closing']['enabled']=True
        with self.assertRaises(ValueError):score_annotation(self.frame(),[True,True],c,'.')

    def test_enabled_head_adds_only_annotation_on_existing_now(self):
        import tempfile
        from unittest.mock import patch,Mock
        from final_solution.tabm_h3.closing import score_annotation
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'closing.joblib').write_bytes(b'fixture');model=Mock();model.predict_proba.return_value=np.array([[.2,.8],[.1,.9]])
            with patch('final_solution.tabm_h3.closing.joblib.load',return_value=model):
                _,_,_,active=score_annotation(self.frame(),[True,False],self.cfg(True),temp)
            np.testing.assert_array_equal(active,[True,False])


class IdempotentRuntimeTests(unittest.TestCase):
    def test_no_new_rows_preserves_disabled_closing_status_without_rescoring(self):
        import tempfile
        from unittest.mock import patch
        from final_solution.tabm_h3.predict import run
        c=config();c['source_paths']={k:'source.csv' for k in f.SOURCE_NAMES};c['initial_state']='state.json';c['closing']={'enabled':False,'status':'diagnostic_only_failed_annual_annotation_gates'}
        _,state=policy.replay(scores(1,date(2026,9,3),25),c)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'source.csv').write_text('fixture');(p/'bundle.json').write_text(json.dumps(c));(p/'state.json').write_text(json.dumps(state))
            panel=pd.DataFrame({'date':[pd.Timestamp('2026-09-03')],'session_ordinal':[25]})
            with patch('final_solution.tabm_h3.predict.read_sources',return_value={}),patch('final_solution.tabm_h3.predict.build_features',return_value=panel),patch('final_solution.tabm_h3.predict.Predictor') as model:
                result=run(p/'bundle.json',p/'out','2026-09-05',mode='historical_smoke')
            model.assert_not_called();self.assertEqual(result['status'],'no_new_source_sessions');self.assertEqual(result['NOW_contacts'],0)
            self.assertEqual(result['closing']['status'],'diagnostic_only_failed_annual_annotation_gates');self.assertEqual(result['closing']['reason_code'],'not_rescored_no_new_sessions');self.assertIsNone(result['closing']['probability']);self.assertFalse(result['authorized_contact'])

if __name__=='__main__':unittest.main()
