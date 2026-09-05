#!/usr/bin/env python3
"""Fetch bounded public historical series for the explicitly requested counterfactual.

FRED/Cboe data are not certified open-license production inputs.  See the
source-specific notices and REPORT.md.  No login, auth bypass or redistribution
is involved.  These frozen files are local research inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, urllib.request, time
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCES = [
    *[(f"treasury_real_{year}", f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value={year}", "treasury") for year in range(2019, 2027)],
    ("treasury_nominal_2019", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2019", "treasury"),
    *[(f"cboe_{s.lower()}", f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{s}_History.csv", "cboe")
      for s in ("VIX", "VIX9D", "VVIX", "OVX", "VXEEM")],
    *[(f"fred_{s.lower()}", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}&cosd=2018-01-01&coed=2026-09-03", "fred")
      for s in ("T10YIE", "T5YIFR", "BAMLH0A0HYM2", "BAMLEMCBPIOAS", "VIXCLS")],
    ("gpr_daily", "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", "gpr"),
    ("gpr_monthly", "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls", "gpr"),
]
NOTICES = [
    ("fred_terms", "https://fred.stlouisfed.org/legal/"),
    ("fred_t10yie_notice", "https://fred.stlouisfed.org/series/T10YIE"),
    ("fred_t5yifr_notice", "https://fred.stlouisfed.org/series/T5YIFR"),
    ("fred_ice_hy_notice", "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"),
    ("fred_ice_em_notice", "https://fred.stlouisfed.org/series/BAMLEMCBPIOAS"),
    ("cboe_history_notice", "https://www.cboe.com/tradable_products/vix/vix_historical_data"),
    ("cboe_terms", "https://www.cboe.com/terms"),
    ("treasury_notice", "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics"),
    ("gpr_notice", "https://www.matteoiacoviello.com/gpr.htm"),
    ("bloomberg_product", "https://professional.bloomberg.com/products/data/data-license/"),
    ("bloomberg_terms", "https://data.bloomberg.com/tos/"),
]

def fetch(sid, url, suffix, family):
    path = ROOT / "raw" / (sid + suffix)
    row = dict(source_id=sid, url=url, family=family, retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaTransferResearch/3.0 (bounded academic feasibility experiment)"})
            with urllib.request.urlopen(req, timeout=45) as res:
                content = res.read()
                row["http_status"] = res.status
                row["content_type"] = res.headers.get("Content-Type", "")
            path.write_bytes(content)
        else:
            row["cache_reused"] = True
        row.update(status="downloaded", path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if sid == "bloomberg_terms":
            row["content_note"]="Raw HTTP response is a JavaScript application shell; substantive terms were separately reviewed through the web tool at the same official URL."
    except Exception as exc:
        row.update(status="unavailable", error=repr(exc))
    print(sid, row["status"], row.get("bytes", row.get("error")), flush=True)
    return row

def main():
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    receipts = []
    for sid, url, family in SOURCES:
        receipts.append(fetch(sid, url, ".xls" if family == "gpr" else (".xml" if family == "treasury" else ".csv"), family))
    for sid, url in NOTICES:
        receipts.append(fetch(sid, url, ".html", "notice"))
    (ROOT / "source_receipt.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
    pd.DataFrame(receipts).to_csv(ROOT / "source_receipt.csv", index=False)

if __name__ == "__main__": main()
