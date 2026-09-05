#!/usr/bin/env python3
"""Extend frozen CNY/RUB spot and fixing history using the existing ISS loader."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO))
from final_solution.data_pipeline import fetch_open_data as old

old.START_DATE="2010-01-01"
old.END_DATE="2019-12-31"
out=HERE / "cny_history"
out.mkdir(exist_ok=True)
receipts=[]
for sid in ("moex_cnyrub_tom","moex_cny_fixing"):
    series=next(s for s in old.MOEX_SERIES if s.series_id==sid)
    frame,receipt,raw=old.fetch_moex(series,workers=2)
    (out / (sid+".json")).write_bytes(raw)
    frame.to_csv(out / (sid+".csv"),index=False)
    receipt["normalized_sha256"]=hashlib.sha256((out / (sid+".csv")).read_bytes()).hexdigest()
    receipt["retrieved_at_utc"]=datetime.now(timezone.utc).isoformat()
    receipts.append(receipt)
    print(sid,len(frame),receipt["first_date"],receipt["last_date"],flush=True)
(out/"source_receipt.json").write_text(json.dumps(receipts,indent=2))
