"""Read-only historical V4 preview. Future outcome columns are never parsed."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,re,sys
from pathlib import Path
from datetime import date
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'final_solution'))
from alphatransfer_final.facts import historical_fact,factual_copy
from alphatransfer_final.product import CURRENCY_COPY

def locate(model):
    if not re.fullmatch(r'[A-Za-z0-9_]+',model):raise ValueError('Invalid registered model identifier')
    path=HERE/'kazakhstan'/f'{model}_predictions.csv.gz'
    receipt=HERE/'kazakhstan'/f'{model}_receipt.json'
    if receipt.exists():record=json.loads(receipt.read_text())
    else:
        path=HERE/'liquidity'/'predictions'/f'{model}.csv.gz'
        records=[]
        for name in ('model_receipts.json','combo_receipts.json'):
            source=HERE/'liquidity'/name
            if source.exists():records.extend(json.loads(source.read_text()))
        record=next((r for r in records if r['name']==model),None)
    if not record or not path.exists():raise ValueError('No completed experiment for this identifier')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get('prediction_sha256')!=digest:raise ValueError('Prediction receipt mismatch')
    return path,digest

def preview(model,as_of,corridor='KZT',policy='legacy',threshold=.5):
    if corridor not in CURRENCY_COPY:raise ValueError('Unknown corridor')
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    if policy not in ('legacy','selective'):raise ValueError('Unknown policy')
    path,digest=locate(model);history=[]
    with gzip.open(path,'rt',newline='') as handle:
        for raw in csv.DictReader(handle):
            if raw['corridor']==corridor and date.fromisoformat(raw['date'])<=as_of:
                # Whitelist: no target, regret or future return enters decisions.
                history.append({'date':raw['date'],'probability':float(raw['probability']),'session':int(raw['session_ordinal']),'stored_candidate':raw['candidate_signal'].lower()=='true'})
    last=-10000;current=None
    for row in sorted(history,key=lambda r:r['date']):
        selected=row['stored_candidate'] if policy=='legacy' else row['probability']>=threshold and row['session']-last>3
        if selected:last=row['session']
        if row['date']==as_of.isoformat():current={**row,'research_candidate':selected}
    if current is None:raise ValueError('No saved OOT decision for this date; no live forecast is fabricated')
    facts=historical_fact(ROOT/'final_solution/data/cbr_daily.csv',corridor,as_of)
    currency,country=CURRENCY_COPY[corridor]
    return {'schema_version':4,'status':'HISTORICAL_RESEARCH_ONLY','model':model,'as_of':as_of.isoformat(),'corridor':corridor,'horizon_effective_cbr_rows':5,'policy':policy,'current':current,'factual_evidence':facts,'factual_copy':factual_copy(facts,currency,country),'prediction_sha256':digest,'eligible_to_send':False,'external_message_sent':False,'interpretation':'Internal probability of no better official reference rate within five subsequent effective observations. Source-price indicators do not constitute an executable Alfa quote. Group policies were evaluated separately on V3 scores; do not silently reuse their thresholds on this model.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--as-of',required=True,type=date.fromisoformat)
    parser.add_argument('--corridor',default='KZT');parser.add_argument('--policy',choices=['legacy','selective'],default='legacy');parser.add_argument('--threshold',default=.5,type=float)
    args=parser.parse_args();print(json.dumps(preview(args.model,args.as_of,args.corridor,args.policy,args.threshold),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
