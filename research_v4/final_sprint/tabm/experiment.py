"""Actual TabM numerical-embedding fits with causal temporal calibration.

All writes stay in this new experiment directory. The 2026 period is an
explicit retrospective benchmark, not an untouched holdout.
"""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,hashlib,importlib.metadata,json,pickle,time,warnings
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from tabm import TabM
from rtdl_num_embeddings import PeriodicEmbeddings
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
from research_v4.continuation.oxr.assess import point
core=e.core;old=e.old
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';CKPT=HERE/'checkpoints'
SEED=20260905
ARCH=dict(n_blocks=2,d_block=128,dropout=.1,k=16)
EMBED=dict(d_embedding=16,n_frequencies=16,frequency_init_scale=.01,activation=True,lite=False)
OPT=dict(lr=.002,weight_decay=.0003)
BASE_SPECS=[dict(scope='pooled',months=m,seed_index=0) for m in (24,60,120)]+[dict(scope='kzt',months=120,seed_index=0)]
MODES={'normal':(24,1),'oxr_delayed':(48,1),'bank_delayed':(24,2),'both_delayed':(48,2)}
CATS={c:i for i,c in enumerate(('AMD','KGS','KZT','TJS','UZS'))}
KEEP=['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date']
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def name(spec):return f"tabm_periodic_{spec['scope']}_{spec['months']}m_s{spec['seed_index']}"
def initialize():
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources=[e.SNAPSHOT,ROOT/'research_v3/models/panel_extended.pkl',ROOT/'research_v4/liquidity/halyk_sell_daily.csv',ROOT/'research_v3/external_data/feature_panel.parquet',Path(e.__file__)]
    protocol=dict(created_unix=time.time(),base_specs=BASE_SPECS,cutoffs=['2025-01-01','2026-01-01'],architecture=ARCH,numerical_embeddings=EMBED,optimizer=OPT,batch_size=256,max_epochs=100,patience=15,selection='Four specifications in 2025, choose lowest KZT Brier of locally calibrated output; two additional seeds of that specification for 2026. No test labels select epochs/calibrator/threshold.',test_status='2026 explicitly retrospective; user permits retrospective choice. No untouched holdout claim.',training='Prior 24/60/120 months before the 12-month calibration year; actual label_available_date strictly before next split. Inner last63 dates, actual maturity purge, train-only preprocessing, full outer train refit at selected epoch.',features='Exact V3 extended BASE, OXR2010 basis+coverage, Halyk six features, Treasury six lag7 features. Add deterministic per-feature missing indicators. Treasury starts2020; no backfill.',calibration='Positive Platt only if prior-year Brier improves; pooled and KZT-only calibration of pooled models. Legacy cooldown/frequency threshold is fit on purged prior-year labels. History outcomes not matured at cutoff are masked in export.',stress='Fixed weights/calibration/threshold and pre-cutoff normal history. OXR D+2 to D+3 and/or Halyk chartdate+1 to+2 after cutoff; recompute rolling basis from stitched causal history. Treasury lag7 unchanged.',ensemble='Equal raw-probability mean of seed0/1/2, separately calibrated on prior year; only selected 2025 specification gets extra seeds.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},code_sha256=sha(__file__),sources=['https://github.com/yandex-research/tabm','https://github.com/yandex-research/rtdl-num-embeddings'])
    if not (HERE/'protocol.json').exists():save(HERE/'protocol.json',protocol)
    save(HERE/'environment.json',dict(python=sys.version,packages={x:importlib.metadata.version(x) for x in ['torch','tabm','rtdl-num-embeddings','numpy','pandas','scikit-learn']},cpu_threads=2,mps_built=torch.backends.mps.is_built(),mps_available=torch.backends.mps.is_available()))
def build():
    views,bankcols=e.build_views()
    # Existing source construction is read-only. Preserve its exact panel contract.
    views={k:augment_panel(v) for k,v in views.items() if k[0]=='2010-01-01'}
    features=e.oxr.BASE+e.oxr.BASIS+e.oxr.COVER+bankcols+TREASURY_LAG7_FEATURES
    return views,features
class Neural:
    def __init__(self,features,seed):self.features=features;self.seed=seed
    def preprocessor(self,frame):
        p=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('gaussian',QuantileTransformer(n_quantiles=min(128,len(frame)),output_distribution='normal',random_state=self.seed))])
        p.fit(frame[self.features]);return p
    def encode(self,frame,pre):
        x=np.concatenate([pre.transform(frame[self.features]),frame[self.features].isna().to_numpy(dtype=float)],axis=1).astype(np.float32)
        assert np.isfinite(x).all()
        return torch.from_numpy(x),torch.from_numpy(frame.corridor.map(CATS).to_numpy(np.int64,copy=True)[:,None])
    def new(self,seed):
        torch.manual_seed(seed);nf=2*len(self.features)
        return TabM.make(n_num_features=nf,cat_cardinalities=[5],d_out=1,num_embeddings=PeriodicEmbeddings(nf,**EMBED),**ARCH)
    @staticmethod
    def predict_encoded(model,encoded):
        model.eval();x,c=encoded;out=[]
        with torch.inference_mode():
            for lo in range(0,len(x),512):out.append(model(x[lo:lo+512],c[lo:lo+512]).squeeze(-1).sigmoid().mean(dim=1).numpy())
        return np.concatenate(out)
    def train(self,frame,pre,seed,epochs,validation=None):
        model=self.new(seed);optim=torch.optim.AdamW(model.parameters(),**OPT)
        x,c=self.encode(frame,pre);y=torch.from_numpy(frame.target.to_numpy(np.float32,copy=True))
        rng=torch.Generator().manual_seed(seed);best=float('inf');bestepoch=1;history=[]
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.
            for idx in torch.randperm(len(x),generator=rng).split(256):
                optim.zero_grad(set_to_none=True);logits=model(x[idx],c[idx]).squeeze(-1)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[idx,None].expand_as(logits))
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();loss_sum+=float(loss.detach())*len(idx)
            row=dict(epoch=epoch,training_bce=loss_sum/len(x))
            if validation is not None:
                vx,vy=validation;p=self.predict_encoded(model,vx);score=float(np.mean((p-vy)**2));row['inner_brier']=score
                if score<best-1e-6:best,bestepoch=score,epoch
            history.append(row)
            if validation is not None and epoch-bestepoch>=15:break
        return model,bestepoch,best,history
    def fit(self,tr,dest):
        clock=time.monotonic();days=sorted(tr.date.unique());inner_start=pd.Timestamp(days[-63])
        innertr=tr[tr.date.lt(inner_start)&tr.label_available_date.lt(inner_start)]
        innerva=tr[tr.date.ge(inner_start)]
        assert innertr.label_available_date.max()<innerva.date.min()
        innerpre=self.preprocessor(innertr)
        _,epochs,score,hist=self.train(innertr,innerpre,self.seed,100,(self.encode(innerva,innerpre),innerva.target.to_numpy(float)))
        self.pre=self.preprocessor(tr)
        self.model,_,_,refit=self.train(tr,self.pre,self.seed+1,epochs)
        dest.mkdir(parents=True,exist_ok=True)
        torch.save(self.model.state_dict(),dest/'weights.pt');joblib.dump(self.pre,dest/'preprocess.joblib')
        pd.DataFrame(hist).to_csv(dest/'inner_epochs.csv',index=False);pd.DataFrame(refit).to_csv(dest/'refit_epochs.csv',index=False)
        meta=dict(seed=self.seed,refit_seed=self.seed+1,selected_epochs=epochs,inner_best_brier=score,inner_train_max=innertr.date.max(),inner_label_max=innertr.label_available_date.max(),inner_validation_min=innerva.date.min(),inner_validation_max=innerva.date.max(),train_rows=len(tr),train_min=tr.date.min(),train_max=tr.date.max(),architecture=ARCH,numerical_embeddings=EMBED,features=self.features,missing_indicators=True,parameters=sum(p.numel() for p in self.model.parameters()),fit_seconds=time.monotonic()-clock,weights_sha256=sha(dest/'weights.pt'),preprocessor_sha256=sha(dest/'preprocess.joblib'))
        save(dest/'model.json',meta);return meta
    def predict(self,frame):return self.predict_encoded(self.model,self.encode(frame,self.pre))
    def load(self,dest):
        self.model=self.new(self.seed+1);self.model.load_state_dict(torch.load(dest/'weights.pt',map_location='cpu',weights_only=True));self.pre=joblib.load(dest/'preprocess.joblib')
def split(views,spec,cutoff):
    p=views['2010-01-01',24,1];end=pd.Timestamp(cutoff.year+1,1,1)
    tr,va,te=old.temporal_split(p,5,cutoff,end,old.Spec(name(spec),months=spec['months'],extended=True))
    cs=cutoff-pd.DateOffset(years=1)
    assert tr.label_available_date.max()<cs and va.label_available_date.max()<cutoff
    history=p[p.date.ge(cs)&p.date.lt(cutoff)].copy()
    if spec['scope']=='kzt':tr,va,te,history=[x[x.corridor.eq('KZT')].copy() for x in (tr,va,te,history)]
    return tr,va,te,history
def calibrate_outputs(raws,spec,cutoff,dest):
    frames=[];calframes=[];policies={}
    for scope in (('pooledcal','kztcal') if spec['scope']=='pooled' else ('kztcal',)):
        group={k:(v if scope=='pooledcal' else v[v.corridor.eq('KZT')]).copy() for k,v in raws.items()}
        va=group['validation'];history=group['history'];cid=name(spec)+'_'+scope
        cal=core.fit_platt_calibrator(va.raw_probability.to_numpy(),va.target)
        vp=core.apply_platt(cal,va.raw_probability.to_numpy());threshold,_,_=core.choose_frequency_threshold(va,vp)
        hp=core.apply_platt(cal,history.raw_probability.to_numpy())
        hi=core.select_per_corridor_with_cooldown(history,hp,threshold)
        state=core.corridor_selection_state(history,hi);ps=core.selection_state(history,core.select_portfolio_from_candidates(history,hp,hi))
        policies[scope]=dict(calibrator=cal,threshold=threshold,initial_state=state,portfolio_state=ps)
        for splitname,p in [('validation',vp),('history',hp)]:
            q=group[splitname].copy();q['probability']=p;q['split']=splitname
            q['config_id']=cid;q['cutoff']=str(cutoff.date());q['fold_test_year']=cutoff.year
            # History includes pre-cutoff decisions with immature outcomes. Those
            # labels are masked so downstream policy code cannot consume them.
            immature=q.label_available_date.ge(cutoff)|q.label_available_date.isna()
            q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
            calframes.append(q)
        for mode in MODES:
            q=group[mode].copy();prob=core.apply_platt(cal,q.raw_probability.to_numpy())
            chosen=core.select_per_corridor_with_cooldown(q,prob,threshold,state)
            portfolio=core.select_portfolio_from_candidates(q,prob,chosen,ps)
            q['probability']=prob;q['candidate_signal']=q.index.isin(chosen);q['signal']=q.index.isin(portfolio)
            q['config_id']=cid;q['cutoff']=str(cutoff.date());q['fold_test_year']=cutoff.year;q['mode']=mode
            q['calibration_scope']=scope;q['training_scope']=spec['scope'];q['months']=spec['months']
            frames.append(q)
    pred=pd.concat(frames,ignore_index=True);cp=pd.concat(calframes,ignore_index=True)
    pred.to_csv(dest.with_suffix('.csv.gz'),index=False);cp.to_csv(dest.with_name(dest.name+'_calibration.csv.gz'),index=False)
    dest.with_name(dest.name+'_policies.pkl').write_bytes(pickle.dumps(policies))
    return pred,cp
def run(views,features,spec,cutoff):
    cutoff=pd.Timestamp(cutoff);stem=name(spec)+'_'+str(cutoff.date());dest=OUT/stem;cd=CKPT/stem
    tr,va,te,history=split(views,spec,cutoff)
    model=Neural(features,SEED+spec['seed_index']*100)
    if (cd/'model.json').exists():model.load(cd);meta=json.loads((cd/'model.json').read_text())
    else:meta=model.fit(tr,cd)
    raw={}
    for key,f in [('validation',va),('history',history)]:
        raw[key]=f[KEEP].copy();raw[key]['raw_probability']=model.predict(f)
    for mode,(delay,lag) in MODES.items():
        q=e.stress_view(views,dict(since='2010-01-01'),cutoff,delay,lag).loc[te.index]
        pd.testing.assert_frame_equal(q[KEEP],te[KEEP],check_exact=True)
        raw[mode]=q[KEEP].copy();raw[mode]['raw_probability']=model.predict(q)
    save(cd/'split.json',dict(cutoff=cutoff,calibration_start=cutoff-pd.DateOffset(years=1),train_min=tr.date.min(),train_max=tr.date.max(),train_latest_label=tr.label_available_date.max(),validation_min=va.date.min(),validation_max=va.date.max(),validation_latest_label=va.label_available_date.max(),train_rows=len(tr),validation_rows=len(va),test_rows=len(te),test_min=te.date.min(),test_max=te.date.max(),oxr_coverage=float(tr.oxr_available.mean()),halyk_coverage=float(tr.halyk_personal_rub_log_price.notna().mean()),treasury_coverage=float(tr[TREASURY_LAG7_FEATURES].notna().all(axis=1).mean()),features_fingerprint=e.fp(tr[['date','corridor',*features]])))
    joblib.dump(raw,cd/'raw_predictions.joblib')
    pred,cp=calibrate_outputs(raw,spec,cutoff,dest)
    save(dest.with_suffix('.json'),dict(spec=spec,cutoff=cutoff,model=meta,protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(__file__),predictions_sha256=sha(dest.with_suffix('.csv.gz')),raw_predictions_sha256=sha(cd/'raw_predictions.joblib')))
    print(stem,'epochs',meta['selected_epochs'],'seconds',round(meta['fit_seconds'],1),'KZT Brier',round(point(pred[pred['mode'].eq('normal')&pred.config_id.str.endswith('kztcal')])['brier'],6),flush=True)
    return pred,cp
def ensemble(spec,cutoff):
    cutoff=pd.Timestamp(cutoff);raws=[]
    for seed in (0,1,2):
        s={**spec,'seed_index':seed};raws.append(joblib.load(CKPT/(name(s)+'_'+str(cutoff.date()))/'raw_predictions.joblib'))
    merged={}
    for key in raws[0]:
        q=raws[0][key].copy()
        for r in raws[1:]:pd.testing.assert_frame_equal(q[KEEP],r[key][KEEP],check_exact=True)
        q['raw_probability']=np.mean([r[key].raw_probability.to_numpy() for r in raws],axis=0);merged[key]=q
    s={**spec,'seed_index':'ensemble3'};return calibrate_outputs(merged,s,cutoff,OUT/(name(s)+'_'+str(cutoff.date())))
def aggregate():
    paths=sorted(OUT.glob('*.csv.gz'));p=[];c=[]
    for path in paths:
        q=pd.read_csv(path,parse_dates=['date','label_available_date'])
        (c if path.name.endswith('_calibration.csv.gz') else p).append(q)
    preds=pd.concat(p,ignore_index=True);cal=pd.concat(c,ignore_index=True)
    preds.to_csv(HERE/'predictions.csv.gz',index=False);cal.to_csv(HERE/'calibration_predictions.csv.gz',index=False)
    rows=[]
    for (year,cid,mode),g in preds.groupby(['fold_test_year','config_id','mode']):
        for scope in (['all','KZT'] if g.corridor.nunique()>1 else ['KZT']):
            q=g if scope=='all' else g[g.corridor.eq('KZT')]
            rows.append(dict(year=year,config_id=cid,mode=mode,scope=scope,**point(q)))
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    return preds,cal
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    initialize()
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,features=build();save(HERE/'features.json',features)
        if args.smoke:
            tr,va,_,_=split(views,BASE_SPECS[0],pd.Timestamp('2025-01-01'))
            m=Neural(features,SEED);pre=m.preprocessor(tr);t=time.monotonic()
            net,_,_,_=m.train(tr,pre,SEED,2)
            print('SMOKE train_rows',len(tr),'features',len(features),'seconds_2epochs',time.monotonic()-t,'parameters',sum(x.numel() for x in net.parameters()),flush=True);return
        dev=[]
        for spec in BASE_SPECS:dev.append(run(views,features,spec,'2025-01-01')[0])
        scores=[]
        for spec,p in zip(BASE_SPECS,dev):scores.append(dict(spec=spec,brier=point(p[p['mode'].eq('normal')&p.config_id.str.endswith('kztcal')])['brier']))
        chosen=min(scores,key=lambda x:x['brier'])['spec'];save(HERE/'selection_2025.json',dict(chosen=chosen,scores=scores,created_unix=time.time(),uses_2026=False))
        for spec in BASE_SPECS:run(views,features,spec,'2026-01-01')
        for seed in (1,2):run(views,features,{**chosen,'seed_index':seed},'2026-01-01')
        ensemble(chosen,'2026-01-01');p,c=aggregate()
        save(HERE/'completion.json',dict(status='complete',neural_fits=10,inner_epoch_selection_fits=10,test_prediction_rows=len(p),calibration_history_rows=len(c),protocol_sha256=sha(HERE/'protocol.json'),selection_sha256=sha(HERE/'selection_2025.json'),code_sha256=sha(__file__)))
if __name__=='__main__':main()
