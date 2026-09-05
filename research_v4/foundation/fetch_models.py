#!/usr/bin/env python3
"""Download public model artifacts; no input data are read or transmitted."""
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HOME','/private/tmp/alphatransfer-hf')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
from huggingface_hub import HfApi, snapshot_download

OUT=Path(__file__).resolve().parent
REVISIONS={'small':'ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a',
           'synth':'3607918a9fd027d5c465d8213e46b98e2c041cea'}
for suffix,revision in REVISIONS.items():
    model_id=f'autogluon/chronos-2-{suffix}'
    info=HfApi().model_info(model_id,revision=revision,files_metadata=True)
    metadata={'id':model_id,'sha':info.sha,'created_at':str(info.created_at),
        'last_modified':str(info.last_modified),'card':info.card_data.to_dict() if info.card_data else {},
        'files':[{'name':s.rfilename,'size':s.size,'blob_id':s.blob_id,
                  'lfs':s.lfs.__dict__ if s.lfs else None} for s in info.siblings]}
    (OUT/f'chronos-2-{suffix}_hub_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(snapshot_download(model_id,revision=revision,
        allow_patterns=['*.json','*.safetensors','README.md','LICENSE']))
