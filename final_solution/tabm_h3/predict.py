"""Source-to-feature-to-NOW H3 CLI, with explicit snapshot/live chronology."""
from __future__ import annotations
from pathlib import Path
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from final_solution.tabm_h3 import TARGET_CONTRACT
    from final_solution.tabm_h3.features import read_sources,build_features,FEATURES
    from final_solution.tabm_h3.model import load_config,Predictor,sha
    from final_solution.tabm_h3.policy import empty_state,validate_state,replay,binding
    from final_solution.tabm_h3.closing import score_annotation
else:
    from . import TARGET_CONTRACT
    from .features import read_sources,build_features,FEATURES
    from .model import load_config,Predictor,sha
    from .policy import empty_state,validate_state,replay,binding
    from .closing import score_annotation
import argparse,json,re
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits


def _json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _scalar(value):
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if hasattr(value,'item'):return value.item()
    return value


def run(bundle_path,output_dir,as_of,*,mode='operational',state_in=None,state_out=None,source_config=None):
    config,base=load_config(bundle_path)
    if mode not in ('operational','historical_smoke'):raise ValueError('Invalid run mode')
    if isinstance(as_of,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',as_of) is None:raise ValueError('run_as_of must be a calendar date YYYY-MM-DD')
    now=pd.Timestamp(as_of)
    if now.tz is not None or now!=now.normalize():raise ValueError('run_as_of must be a naive calendar date')
    cutoff=pd.Timestamp(config['model_cutoff']).normalize()
    if now<cutoff:raise ValueError('Run as_of precedes the model availability cutoff')
    paths=config['source_paths'] if source_config is None else json.loads(Path(source_config).read_text())
    source_sha={k:sha(base/Path(v)) for k,v in paths.items()}
    if source_config is None and config.get('source_sha256') and source_sha!=config['source_sha256']:raise ValueError('Frozen source checksum mismatch')
    sources=read_sources(paths,base);features=build_features(sources,now)
    initial=Path(state_in) if state_in else base/config['initial_state']
    if not initial.is_file():raise ValueError('Explicit initial score/candidate state is required')
    state=json.loads(initial.read_text());validate_state(state,config)
    if state['last_processed_date'] and pd.Timestamp(state['last_processed_date'])>now:raise ValueError('State is from the future relative to run_as_of')
    fresh=features if state['last_processed_session'] is None else features[features.session_ordinal.gt(state['last_processed_session'])]
    latest=features.iloc[-1];output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    common={'profile_id':config['profile_id'],'model_id':config['model_id'],'model_cutoff':config['model_cutoff'],'train_horizon':3,'target_contract':TARGET_CONTRACT,'scenario':'NOW','corridor':'KZT','run_as_of':now.date().isoformat(),'latest_source_feature_date':latest.date.date().isoformat(),'source_lag_calendar_days':int((now-latest.date).days),'mode':mode,'authorized_contact':False,'external_messages_sent':0,'closing':{'status':config.get('closing',{}).get('status','unavailable'),'enabled':bool(config.get('closing',{}).get('enabled',False)),'reason':'Not evaluated until a new source row is processed','probability':None},'source_sha256':source_sha,'state_binding_sha256':binding(config),'limitations':['CBR reference target; executable Alpha quote is not supplied.','Known-at clocks for CBR/MOEX/Halyk/Treasury follow the documented inherited timing assumptions.']}
    if fresh.empty:
        common['closing']['reason_code']='not_rescored_no_new_sessions'
        result={**common,'status':'no_new_source_sessions','NOW_contacts':0,'rows':0,'candidate_signal':False,'reason_codes':['already_processed_source_prefix']}
        _json(output/'signal_decision.json',result)
        if state_out:_json(state_out,state)
        return result
    if mode=='operational' and fresh.date.lt(cutoff).any():raise ValueError('Pre-cutoff rows require explicit historical_smoke; they are not operational model forecasts')
    if mode=='operational' and latest.date<now:
        # A stale reference feature snapshot can still be inspected in smoke mode,
        # but does not claim a fresh model-driven operational action today.
        raise ValueError('No current-day feature observation; use historical_smoke for a dated snapshot')
    if state['last_processed_session'] is not None and int(fresh.session_ordinal.min())!=state['last_processed_session']+1:raise ValueError('Source prefix cannot continue this state without a missing session')
    predictor=Predictor(config,base);raw,prob=predictor.predict(fresh)
    score_rows=[{'date':r.date.date().isoformat(),'corridor':'KZT','session_ordinal':int(r.session_ordinal),'probability':float(p)} for r,p in zip(fresh.itertuples(),prob)]
    decisions,next_state=replay(score_rows,config,state)
    result=pd.DataFrame(decisions);result['raw_probability']=raw;result['probability']=prob;result['train_horizon']=3;result['target_contract']=TARGET_CONTRACT
    result['model_id']=config['model_id'];result['model_cutoff']=config['model_cutoff'];result['run_as_of']=now.date().isoformat();result['mode']=mode
    result['chronology']=np.where(fresh.date.to_numpy()<cutoff.to_datetime64(),'pre_model_cutoff_snapshot','post_model_cutoff')
    result['primary_scenario']=np.where(result.candidate_signal,'NOW','NONE');result['model_verdict']=np.where(result.candidate_signal,'Модельный сигнал: выгодно сейчас','Нет модельного сигнала NOW')
    result['rub_per_kzt_reference']=fresh.rub_per_unit.to_numpy();result['observed_ret1']=fresh.ret1.to_numpy();result['historical_rank60']=fresh.pr60.to_numpy()
    result['feature_known_at']=[str(x) for x in fresh.feature_known_at];result['authorized_contact']=False
    closing_meta,closing_prob,closing_condition,closing_annotation=score_annotation(fresh,result.candidate_signal.to_numpy(),config,base)
    if closing_prob is not None:
        result['closing_probability']=closing_prob;result['closing_diagnostic_condition']=closing_condition;result['closing_annotation']=closing_annotation;result['closing_train_horizon']=3
        result.loc[closing_annotation,'model_verdict']+='; есть модельные признаки закрытия окна'
    result.to_csv(output/'predictions.csv',index=False)
    # Exposed latest decision carries facts, own horizon and reasons, never labels.
    last=result.iloc[-1];snapshot=last.chronology=='pre_model_cutoff_snapshot'
    decision={**common,'status':'historical_snapshot_smoke' if mode=='historical_smoke' else 'operational_model_candidate','rows':len(result),'NOW_contacts':int(result.candidate_signal.sum()),'feature_date':str(last.date),'feature_known_at':str(last.feature_known_at),'chronology':'in_calibration_or_training_snapshot' if snapshot else 'post_model_cutoff','candidate_signal':bool(last.candidate_signal),'probability':float(last.probability),'raw_probability':float(last.raw_probability),'rank_score':_scalar(last.rank_score),'prior_score_count':int(last.prior_score_count),'threshold':float(last.threshold),'reason_codes':list(last.reason_codes),'model_verdict':str(last.model_verdict),'factual_context':{'rub_per_kzt_reference':float(last.rub_per_kzt_reference),'observed_ret1':_scalar(last.observed_ret1),'historical_rank60':_scalar(last.historical_rank60)},'probability_is_h3':True,'is_oot_claim':False if mode=='historical_smoke' else None,'preview_text':f'Опубликованный справочный курс на {last.date}: {last.rub_per_kzt_reference:.6f} RUB за 1 KZT. Актуальная котировка перевода рассчитывается отдельно.'}
    if snapshot and config.get('metadata',{}).get('training',{}).get('calibration_start'):
        calstart=pd.Timestamp(config['metadata']['training']['calibration_start'])
        decision['chronology']='in_calibration_snapshot' if pd.Timestamp(last.date)>=calstart else 'in_training_snapshot'
    decision['closing']={**closing_meta,'probability':float(closing_prob[-1]) if closing_prob is not None else None,'diagnostic_condition':bool(closing_condition[-1]) if closing_condition is not None else False,'annotation_active':bool(closing_annotation[-1]) if closing_annotation is not None else False}
    decision['annotations']=[{'scenario':'CLOSING','train_horizon':3,'probability':float(closing_prob[-1]),'target_contract':closing_meta['target_contract']}] if closing_annotation is not None and closing_annotation[-1] else []
    _json(output/'signal_decision.json',decision);_json(output/'next_state.json',next_state)
    if state_out:_json(state_out,next_state)
    _json(output/'run_receipt.json',{'bundle_sha256':sha(bundle_path),'source_sha256':source_sha,'state_input_sha256':sha(initial),'predictions_sha256':sha(output/'predictions.csv'),'next_state_sha256':sha(output/'next_state.json'),'state_binding_sha256':binding(config),'rows':len(result),'run_as_of':now.date().isoformat(),'mode':mode,'research_imports_required':False})
    return decision


def main(argv=None):
    parser=argparse.ArgumentParser(description='Standalone KZT NOW H3 TabM runtime')
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent/'bundle.json');parser.add_argument('--as-of')
    parser.add_argument('--mode',choices=['operational','historical_smoke'],default=None);parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--state-in',type=Path);parser.add_argument('--state-out',type=Path);parser.add_argument('--sources',type=Path)
    a=parser.parse_args(argv)
    cfg,_=load_config(a.bundle);as_of=a.as_of or cfg.get('default_as_of',cfg['model_cutoff']);mode=a.mode or cfg.get('metadata',{}).get('default_mode','operational')
    with threadpool_limits(limits=1):result=run(a.bundle,a.output_dir,as_of,mode=mode,state_in=a.state_in,state_out=a.state_out,source_config=a.sources)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
