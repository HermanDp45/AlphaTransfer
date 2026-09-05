"""Inference-only official TabM loader; no training/research module dependency."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import joblib,numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_config(config):
    from .features import FEATURES
    if config.get('schema_version')!=1 or config.get('train_horizon')!=3 or config.get('corridor')!='KZT':raise ValueError('Only schema1 KZT NOW H3 is supported')
    if config.get('features')!=FEATURES:raise ValueError('Feature names/order differ from FULL33 contract')
    cal=config['calibration']
    if cal['method']!='identity' and (cal['method']!='prior_year_monotone_platt' or not np.isfinite([cal['intercept'],cal['slope']]).all() or cal['slope']<=0):raise ValueError('Invalid monotone calibration')
    return config


def load_config(path):
    path=Path(path);config=validate_config(json.loads(path.read_text()));return config,path.parent


class Predictor:
    def __init__(self,config,base):
        import torch
        from tabm import TabM
        from rtdl_num_embeddings import PeriodicEmbeddings
        self.config=validate_config(config);self.base=Path(base);self.features=config['features']
        torch.set_num_threads(1)
        weights=self.base/config['model']['weights'];pre=self.base/config['model']['preprocessor']
        if sha(weights)!=config['model']['weights_sha256'] or sha(pre)!=config['model']['preprocessor_sha256']:raise ValueError('Model artifact checksum mismatch')
        self.pre=joblib.load(pre)
        nf=2*len(self.features)
        self.model=TabM.make(n_num_features=nf,cat_cardinalities=[5],d_out=1,num_embeddings=PeriodicEmbeddings(nf,**config['numerical_embeddings']),**config['architecture'])
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));self.model.eval()

    def predict(self,frame):
        import torch
        if not frame.corridor.eq('KZT').all():raise ValueError('KZT-only model cannot score other corridors')
        missing=set(self.features)-set(frame)
        if missing:raise ValueError('Missing features '+str(sorted(missing)))
        x=np.concatenate([self.pre.transform(frame[self.features]),frame[self.features].isna().to_numpy(float)],axis=1).astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite transformed features')
        # The official model uses five category codes; KZT is code2 even when
        # weights were fitted on KZT-only samples. Native encoding is one-hot.
        c=np.full((len(x),1),2,dtype=np.int64);parts=[]
        with torch.inference_mode():
            for start in range(0,len(x),512):parts.append(self.model(torch.from_numpy(x[start:start+512]),torch.from_numpy(c[start:start+512])).squeeze(-1).sigmoid().mean(dim=1).numpy())
        raw=np.concatenate(parts).astype(np.float64) if parts else np.array([],dtype=np.float64)
        cal=self.config['calibration']
        if cal['method']=='identity':prob=raw.copy()
        else:
            from scipy.special import expit
            clipped=np.clip(raw.astype(np.float64),1e-6,1-1e-6);z=cal['intercept']+cal['slope']*np.log(clipped/(1-clipped));prob=expit(z)
        return raw,prob
