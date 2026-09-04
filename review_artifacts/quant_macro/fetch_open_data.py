#!/usr/bin/env python3
"""Download and normalize a compact, license-audited exogenous data bundle.

Only first-party sources are queried.  FRED is deliberately excluded: its
current terms prohibit using FRED content to develop or train ML systems.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import xlrd
from bs4 import BeautifulSoup


START_DATE = "2020-01-01"
END_DATE = "2026-09-03"
USER_AGENT = "AlphaTransferResearch/1.0 (reproducible hackathon research)"
MOEX_DATA_POLICY_URL = "https://www.moex.com/en/datapolicy/"
MOEX_QUOTE_PARAMETER_SPEC_URL = "https://www.moex.com/files/4q0ffn3fx8pas77s2dfzm78j4n"


@dataclass(frozen=True)
class MoexSeries:
    series_id: str
    engine: str
    market: str
    board: str | None
    security: str
    columns: tuple[str, ...]
    quote_units: int | None = None


MOEX_SERIES = (
    MoexSeries(
        "moex_cnyrub_tom",
        "currency",
        "selt",
        "CETS",
        "CNYRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        1,
    ),
    MoexSeries(
        "moex_kztrub_tom",
        "currency",
        "selt",
        "CETS",
        "KZTRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        100,
    ),
    MoexSeries(
        "moex_amdrub_tom",
        "currency",
        "selt",
        "CETS",
        "AMDRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        100,
    ),
    MoexSeries(
        "moex_kgsrub_tom",
        "currency",
        "selt",
        "CETS",
        "KGSRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        100,
    ),
    MoexSeries(
        "moex_uzsrub_tom",
        "currency",
        "selt",
        "CETS",
        "UZSRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        10_000,
    ),
    MoexSeries(
        "moex_tjsrub_tom",
        "currency",
        "selt",
        "CETS",
        "TJSRUB_TOM",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "NUMTRADES", "WAPRICE"),
        10,
    ),
    MoexSeries(
        "moex_imoex",
        "stock",
        "index",
        "SNDX",
        "IMOEX",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "VALUE", "VOLUME"),
    ),
    MoexSeries(
        "moex_rgbitr",
        "stock",
        "index",
        "SNDX",
        "RGBITR",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "VALUE", "VOLUME"),
    ),
    MoexSeries(
        "moex_rvi",
        "stock",
        "index",
        "RTSI",
        "RVI",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE", "VALUE", "VOLUME"),
    ),
    MoexSeries(
        "moex_cny_fixing",
        "currency",
        "index",
        None,
        "CNYFIXME",
        ("TRADEDATE", "SECID", "CLOSE"),
    ),
    MoexSeries(
        "moex_rusfar",
        "stock",
        "index",
        "MMIX",
        "RUSFAR",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE"),
    ),
    MoexSeries(
        "moex_rusfar_cny",
        "stock",
        "index",
        "MMIX",
        "RUSFARCNY",
        ("TRADEDATE", "OPEN", "LOW", "HIGH", "CLOSE"),
    ),
)

FED_H10_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H10&series=122e3bcb627e8e53f1bf72a1a09cfb81&lastobs="
    "&from=01/01/2020&to=09/03/2026&filetype=csv&label=include&layout=seriescolumn"
)
TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?"
    "data=daily_treasury_yield_curve&field_tdr_date_value={year}"
)
EIA_BRENT_URL = "https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls"
WORLD_BANK_COMMODITIES_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
)
KZ_CPI_URL = "https://stat.gov.kz/api/iblock/element/14511/file/ru/"
KZ_DATA_USE_URL = "https://stat.gov.kz/ru/description/"
NBK_OPEN_DATA_URL = "https://data.nationalbank.kz/api/v1/data"
NBK_INDICATORS_URL = "https://data.nationalbank.kz/api/v1/data/indicators"
NBK_DATA_TERMS_URL = "https://nationalbank.kz/en/page/data-usage-terms"
NBK_OPEN_DATA_ANNOUNCEMENT_URL = (
    "https://nationalbank.kz/en/news/informacionnye-soobshcheniya/17144"
)
CBR_RESERVES_URL = (
    "https://www.cbr.ru/eng/hd_base/mrrf/mrrf_7d/?UniDbQuery.Posted=True"
    "&UniDbQuery.From=01.2020&UniDbQuery.To=08.2026"
)
CBR_BUSINESS_CLIMATE_URL = (
    "https://www.cbr.ru/dataservice/data?y1=2020&y2=2026"
    "&datasetId=58&publicationId=25&measureId=119"
)
CBR_CURRENT_ACCOUNT_URL = (
    "https://www.cbr.ru/dataservice/data?y1=2020&y2=2026"
    "&datasetId=9&publicationId=8"
)
NBK_BUSINESS_ACTIVITY_URL = "https://nationalbank.kz/file/download/134770"
NBK_INFLATION_EXPECTATIONS_URL = "https://nationalbank.kz/file/download/133154"
ECB_CISS_URLS = {
    "ecb_ciss_euro_area": (
        "https://data-api.ecb.europa.eu/service/data/CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX?"
        "format=csvdata&startPeriod=2020-01-01&endPeriod=2026-09-03"
    ),
    "ecb_ciss_us": (
        "https://data-api.ecb.europa.eu/service/data/CISS/D.US.Z0Z.4F.EC.SS_CIN.IDX?"
        "format=csvdata&startPeriod=2020-01-01&endPeriod=2026-09-03"
    ),
    "ecb_ciss_fx": (
        "https://data-api.ecb.europa.eu/service/data/CISS/D.U2.Z0Z.4F.EC.SS_FXN.CON?"
        "format=csvdata&startPeriod=2020-01-01&endPeriod=2026-09-03"
    ),
}
CBR_KEY_RATE_URL = (
    "https://www.cbr.ru/eng/hd_base/KeyRate/?UniDbQuery.Posted=True"
    "&UniDbQuery.From=01.01.2020&UniDbQuery.To=03.09.2026"
)
NBK_RATE_RUBRICS = {
    2020: 1543,
    2021: 1581,
    2022: 1698,
    2023: 1843,
    2024: 2098,
    2025: 2237,
    2026: 2365,
}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[2])
    parser.add_argument("--output-dir", type=Path, default=script.parent)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, attempts: int = 4) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower() in {"content-type", "etag", "last-modified"}
                }
                return response.read(), headers
        except Exception as error:  # noqa: BLE001 - retry heterogeneous network failures
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def moex_url(series: MoexSeries, start: int) -> str:
    columns = ",".join(series.columns)
    board_path = f"boards/{series.board}/" if series.board else ""
    return (
        f"https://iss.moex.com/iss/history/engines/{series.engine}/markets/{series.market}/"
        f"{board_path}securities/{series.security}.json?"
        f"from={START_DATE}&till={END_DATE}&start={start}&iss.meta=off"
        f"&iss.only=history,history.cursor&history.columns={columns}"
    )


def decode_moex_page(content: bytes) -> tuple[list[str], list[list[Any]], tuple[int, int, int]]:
    payload = json.loads(content)
    history = payload["history"]
    cursor_rows = payload.get("history.cursor", {}).get("data", [])
    if cursor_rows:
        cursor = tuple(int(value) for value in cursor_rows[0])
    else:
        cursor = (0, len(history["data"]), 100)
    return history["columns"], history["data"], cursor


def freshness_metadata(
    frame: pd.DataFrame,
    date_column: str,
    maximum_staleness_days: int,
    fail_closed: bool = True,
) -> dict[str, Any]:
    last_date = pd.to_datetime(frame[date_column], errors="coerce").max()
    reference_date = pd.Timestamp(END_DATE)
    staleness_days = int((reference_date - last_date.normalize()).days)
    status = "pass" if staleness_days <= maximum_staleness_days else "fail"
    if status == "fail" and fail_closed:
        raise ValueError(
            f"{date_column}: source is {staleness_days} days stale; "
            f"maximum is {maximum_staleness_days}"
        )
    return {
        "reference_date": END_DATE,
        "observed_last_date": str(last_date.date()),
        "staleness_days": staleness_days,
        "maximum_staleness_days": maximum_staleness_days,
        "status": status,
    }


def fetch_moex(series: MoexSeries, workers: int) -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    first_content, first_headers = fetch(moex_url(series, 0))
    columns, first_rows, cursor = decode_moex_page(first_content)
    _, total, page_size = cursor
    pages: dict[int, list[list[Any]]] = {0: first_rows}
    starts = list(range(page_size, total, page_size))

    def one_page(start: int) -> tuple[int, list[list[Any]]]:
        content, _ = fetch(moex_url(series, start))
        page_columns, rows, _ = decode_moex_page(content)
        if page_columns != columns:
            raise ValueError(f"MOEX schema drift for {series.series_id}")
        return start, rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_page, start) for start in starts]
        for future in as_completed(futures):
            start, rows = future.result()
            pages[start] = rows

    all_rows = [row for start in sorted(pages) for row in pages[start]]
    frame = pd.DataFrame(all_rows, columns=columns)
    if "TRADEDATE" not in frame or frame.empty:
        raise ValueError(f"empty or malformed MOEX series {series.series_id}")
    frame = frame.drop_duplicates("TRADEDATE").sort_values("TRADEDATE")
    active = frame[pd.to_numeric(frame["CLOSE"], errors="coerce").gt(0)]
    if series.quote_units is not None:
        for column in ("OPEN", "LOW", "HIGH", "CLOSE", "WAPRICE"):
            if column in frame:
                frame[f"{column}_RUB_PER_UNIT"] = (
                    pd.to_numeric(frame[column], errors="coerce") / series.quote_units
                )
    assembled_raw = json.dumps(
        {"columns": columns, "data": all_rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    metadata = {
        "source_id": series.series_id,
        "authority": "Moscow Exchange (MOEX ISS)",
        "url_template": moex_url(series, 0).replace("start=0", "start={offset}"),
        "license_review": "conditional; validate MOEX market-data user agreement before production",
        "terms_url": MOEX_DATA_POLICY_URL,
        "quote_parameter_spec_url": MOEX_QUOTE_PARAMETER_SPEC_URL,
        "availability": "daily close; experiment applies a conservative lag",
        "quote_units": series.quote_units,
        "normalized_price_unit": (
            "RUB_per_one_base_currency_unit" if series.quote_units is not None else None
        ),
        "rows": len(frame),
        "first_date": str(frame["TRADEDATE"].min()),
        "last_date": str(frame["TRADEDATE"].max()),
        "freshness": freshness_metadata(frame, "TRADEDATE", 7),
        "active_market_rows": len(active),
        "active_market_row_rate": len(active) / len(frame),
        "active_market_freshness": (
            freshness_metadata(active, "TRADEDATE", 14, fail_closed=False)
            if not active.empty
            else {"status": "fail", "reason": "no positive close observations"}
        ),
        "assembled_raw_sha256": sha256_bytes(assembled_raw),
        "response_headers": first_headers,
    }
    return frame, metadata, assembled_raw


def normalize_fed_h10(content: bytes) -> pd.DataFrame:
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    identifiers = rows[4][1:]
    columns = ["date", *identifiers]
    data = rows[6:]
    frame = pd.DataFrame(data, columns=columns).replace({"ND": math.nan, "": math.nan})
    for column in identifiers:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date")


def normalize_treasury(contents: dict[int, bytes]) -> pd.DataFrame:
    data_namespace = "http://schemas.microsoft.com/ado/2007/08/dataservices"
    rows: list[dict[str, Any]] = []
    for year in sorted(contents):
        root = ET.fromstring(contents[year])
        for properties in root.findall(".//{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}properties"):
            values = {child.tag.split("}")[-1]: child.text for child in properties}
            rows.append(
                {
                    "date": values.get("NEW_DATE", "")[:10],
                    "us_2y": values.get("BC_2YEAR"),
                    "us_10y": values.get("BC_10YEAR"),
                    "us_30y": values.get("BC_30YEAR"),
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    for column in ("us_2y", "us_10y", "us_30y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty or not data_namespace:
        raise ValueError("empty Treasury curve")
    return frame


def normalize_eia_brent(content: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(content), sheet_name="Data 1", header=None, skiprows=3)
    frame = frame.iloc[:, :2]
    frame.columns = ["date", "brent_usd_per_barrel"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["brent_usd_per_barrel"] = pd.to_numeric(frame["brent_usd_per_barrel"], errors="coerce")
    return frame.dropna().query("date >= @START_DATE and date <= @END_DATE").sort_values("date")


def normalize_world_bank(content: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(content), sheet_name="Monthly Prices", header=None)
    names = frame.iloc[4].tolist()
    selected = {
        "Crude oil, Brent": "wb_brent",
        "Natural gas, Europe": "wb_europe_gas",
        "Wheat, US SRW": "wb_wheat",
        "Urea ": "wb_urea",
        "Aluminum": "wb_aluminum",
        "Copper": "wb_copper",
        "Nickel": "wb_nickel",
        "Zinc": "wb_zinc",
        "Gold": "wb_gold",
    }
    indexes = {new: names.index(old) for old, new in selected.items()}
    result = pd.DataFrame({"period": frame.iloc[6:, 0].astype(str)})
    for new_name, index in indexes.items():
        result[new_name] = pd.to_numeric(frame.iloc[6:, index], errors="coerce")
    parsed = pd.to_datetime(result["period"].str.replace("M", "-", regex=False) + "-01", errors="coerce")
    result["period_start"] = parsed
    result["period_end"] = parsed + pd.offsets.MonthEnd(0)
    # World Bank says the file is updated on the second business day. Seven
    # calendar days is a conservative operational proxy, but not a vintage.
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=7)
    return result.dropna(subset=["period_start"]).sort_values("period_start")


def normalize_kz_cpi(content: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(content), sheet_name="ИПЦ", header=None)
    month_numbers = {
        "Январь": 1,
        "Февраль": 2,
        "Март": 3,
        "Апрель": 4,
        "Май": 5,
        "Июнь": 6,
        "Июль": 7,
        "Август": 8,
        "Сентябрь": 9,
        "Октябрь": 10,
        "Ноябрь": 11,
        "Декабрь": 12,
    }
    rows: list[dict[str, Any]] = []
    current_year: int | None = None
    for _, row in frame.iterrows():
        label = str(row.iloc[0]).strip()
        if re.fullmatch(r"20\d{2}", label):
            current_year = int(label)
            continue
        if current_year is None or label not in month_numbers:
            continue
        month = month_numbers[label]
        rows.append(
            {
                "period_start": date(current_year, month, 1),
                "kz_cpi_mom_index": pd.to_numeric(row.iloc[1], errors="coerce"),
                "kz_cpi_yoy_index": pd.to_numeric(row.iloc[9], errors="coerce"),
            }
        )
    result = pd.DataFrame(rows).dropna(subset=["kz_cpi_mom_index"])
    result["period_start"] = pd.to_datetime(result["period_start"])
    result["period_end"] = result["period_start"] + pd.offsets.MonthEnd(0)
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=7)
    return result.sort_values("period_start")


def normalize_nbk_open_data(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(content)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("NBK open-data response has an unexpected schema")
    frame = pd.DataFrame(payload["rows"])
    if frame.empty or "report_date" not in frame:
        raise ValueError("NBK open-data response is empty")
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["report_date", "amount"]).sort_values("report_date")
    metadata = {
        "form_id": payload.get("formId"),
        "total_rows_reported": payload.get("totalRows"),
        "columns": payload.get("columns"),
    }
    return frame, metadata


def normalize_nbk_reserves(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = normalize_nbk_open_data(content)
    labels = frame.get("class_type", pd.Series(index=frame.index, dtype=object)).fillna(
        frame.get("subtype", pd.Series(index=frame.index, dtype=object))
    )
    labels = labels.astype(str).str.strip().str.casefold()
    mapping = {
        "gross international reserves": "kz_gross_reserves_usd_mn",
        "net international reserves": "kz_net_reserves_usd_mn",
        "foreign currency assets of the national fund of kazakhstan": "kz_national_fund_fx_assets_usd_mn",
    }
    selected = frame.assign(series=labels.map(mapping)).dropna(subset=["series"])
    result = selected.pivot_table(
        index="report_date",
        columns="series",
        values="amount",
        aggfunc="last",
    ).reset_index()
    required = set(mapping.values())
    if missing := required - set(result.columns):
        raise ValueError(f"NBK reserves response is missing series {sorted(missing)}")
    result["period_end"] = result["report_date"] + pd.offsets.MonthEnd(0)
    # The release calendar distinguishes preliminary and adjusted estimates.
    # Forty calendar days after period-end is conservative for this eval-only snapshot.
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=40)
    return result.sort_values("report_date"), metadata


def normalize_nbk_current_account(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = normalize_nbk_open_data(content)
    code = frame.get("code", pd.Series(index=frame.index, dtype=object)).astype(str)
    account = frame.get("account_type_code", pd.Series(index=frame.index, dtype=object)).astype(str)
    selected = frame[
        code.str.strip().str.casefold().eq("current account")
        & account.str.strip().str.casefold().eq("current account")
    ].copy()
    if selected.empty:
        raise ValueError("NBK current-account row is absent")
    result = selected.groupby("report_date", as_index=False)["amount"].last()
    result = result.rename(columns={"amount": "kz_current_account_usd_mn"})
    result["period_end"] = result["report_date"] + pd.offsets.QuarterEnd(0)
    # Quarterly BOP estimates arrive roughly three months after quarter-end.
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=100)
    return result.sort_values("report_date"), metadata


def normalize_nbk_kase_monthly(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = normalize_nbk_open_data(content)
    rub = frame[frame["currency"].astype(str).str.strip().str.casefold().eq("russian ruble")].copy()
    labels = rub["type"].astype(str).str.strip().str.casefold()
    mapping = {
        "volume of trade on kase for the period (units of currency)": "kase_rub_volume_units",
        "average exchange rate on kase end of period (tenge)": "kase_rubkzt_average_eop",
    }
    rub["series"] = labels.map(mapping)
    result = rub.dropna(subset=["series"]).pivot_table(
        index="report_date",
        columns="series",
        values="amount",
        aggfunc="last",
    ).reset_index()
    if result.empty:
        raise ValueError("NBK KASE monthly RUB aggregate is absent")
    result["period_end"] = result["report_date"] + pd.offsets.MonthEnd(0)
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=40)
    return result.sort_values("report_date"), metadata


def normalize_nbk_interbank_monthly(content: bytes) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, metadata = normalize_nbk_open_data(content)
    # Form 181 currently omits the category key that distinguishes multiple
    # rates within the same currency/month. Preserve every row rather than
    # silently collapsing economically different observations.
    duplicate_key = frame.duplicated(["report_date", "currency", "type"], keep=False)
    frame["ambiguous_duplicate_key"] = duplicate_key
    return frame.sort_values(["report_date", "currency", "type", "amount"]), metadata


def normalize_nbk_indicators(content: bytes) -> pd.DataFrame:
    payload = json.loads(content)
    if not isinstance(payload, list) or not payload:
        raise ValueError("NBK indicators response is empty or malformed")
    frame = pd.DataFrame(payload)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    value_columns = ["inflationTarget", "annualInflation", "baseRate", "tonia"]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


def normalize_cbr_reserves(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    rows: list[dict[str, Any]] = []
    for tr in soup.select("table.data tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.select("td")]
        if len(cells) != 2 or not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", cells[0]):
            continue
        rows.append(
            {
                "date": datetime.strptime(cells[0], "%d.%m.%Y").date(),
                "ru_international_reserves_usd_bn": float(cells[1].replace(",", "")),
            }
        )
    result = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    if result.empty:
        raise ValueError("CBR international-reserves table is empty")
    return result


def normalize_cbr_business_climate(content: bytes) -> pd.DataFrame:
    payload = json.loads(content)
    headers = {int(item["id"]): str(item["elname"]) for item in payload.get("headerData", [])}
    expected_headers = {
        41: "Индикатор бизнес-климата Банка России",
        42: "Индикатор бизнес-климата Банка России (факт.)",
        43: "Индикатор бизнес-климата Банка России (ожид.)",
        48: "Ценовые ожидания предприятий на следующие 3 месяца (баланс ответов)",
    }
    for column_id, expected in expected_headers.items():
        if headers.get(column_id) != expected:
            raise ValueError(f"CBR business-climate schema drift for colId={column_id}")
    raw = pd.DataFrame(payload.get("RawData", []))
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["obs_val"] = pd.to_numeric(raw["obs_val"], errors="coerce")
    raw = raw[raw["colId"].isin(expected_headers)].dropna(subset=["date", "obs_val"])
    result = raw.pivot_table(index="date", columns="colId", values="obs_val", aggfunc="last")
    result = result.rename(
        columns={
            41: "ru_business_climate",
            42: "ru_business_climate_current",
            43: "ru_business_climate_expectations",
            48: "ru_business_price_expectations",
        }
    ).reset_index()
    required = {
        "ru_business_climate",
        "ru_business_climate_current",
        "ru_business_climate_expectations",
        "ru_business_price_expectations",
    }
    if missing := required - set(result.columns):
        raise ValueError(f"CBR business-climate response is missing {sorted(missing)}")
    # API date is the first day after the referenced month. The public release
    # arrives later, so the experiment waits another 14 calendar days.
    result["period_end"] = result["date"] - pd.Timedelta(days=1)
    result["available_date_proxy"] = result["date"] + pd.Timedelta(days=14)
    return result.sort_values("period_end")


def normalize_cbr_current_account(content: bytes) -> pd.DataFrame:
    payload = json.loads(content)
    headers = {int(item["id"]): str(item["elname"]) for item in payload.get("headerData", [])}
    if headers.get(-1) != "Сальдо счета текущих операций":
        raise ValueError("CBR current-account schema drift")
    raw = pd.DataFrame(payload.get("RawData", []))
    result = pd.DataFrame(
        {
            "period_marker": pd.to_datetime(raw["date"], errors="coerce"),
            "ru_current_account_usd_mn": pd.to_numeric(raw["obs_val"], errors="coerce"),
        }
    ).dropna()
    result["period_end"] = result["period_marker"] - pd.Timedelta(days=1)
    result["available_date_proxy"] = result["period_marker"] + pd.Timedelta(days=90)
    return result.drop_duplicates("period_end").sort_values("period_end")


def normalize_nbk_business_activity(content: bytes) -> pd.DataFrame:
    frame = pd.read_excel(
        io.BytesIO(content),
        sheet_name="Data series",
        header=None,
        skiprows=2,
        usecols=range(7),
    )
    frame.columns = [
        "period",
        "kz_bai_economy",
        "kz_bai_production",
        "kz_bai_service",
        "kz_bai_construction",
        "kz_bai_mining",
        "kz_bai_trade",
    ]
    parsed = frame["period"].astype(str).str.extract(r"^(20\d{2})_(\d{1,2})$")
    valid = parsed.notna().all(axis=1)
    frame = frame.loc[valid].copy()
    frame["period_start"] = pd.to_datetime(
        parsed.loc[valid, 0] + "-" + parsed.loc[valid, 1].str.zfill(2) + "-01"
    )
    value_columns = [column for column in frame if column.startswith("kz_bai_")]
    for column in value_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["kz_bai_sector_share_above_50"] = frame[value_columns[1:]].gt(50.0).mean(axis=1)
    frame["kz_bai_sector_dispersion"] = frame[value_columns[1:]].std(axis=1)
    frame["period_end"] = frame["period_start"] + pd.offsets.MonthEnd(0)
    frame["available_date_proxy"] = frame["period_end"] + pd.Timedelta(days=7)
    keep = [
        "period_start",
        "period_end",
        "available_date_proxy",
        *value_columns,
        "kz_bai_sector_share_above_50",
        "kz_bai_sector_dispersion",
    ]
    return frame[keep].dropna(subset=["kz_bai_economy"]).sort_values("period_start")


def normalize_nbk_inflation_expectations(content: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(content), sheet_name="Medians", header=None, skiprows=4)
    result = pd.DataFrame(
        {
            "period_start": pd.to_datetime(frame.iloc[:, 0], errors="coerce"),
            "kz_perceived_inflation_12m_pct": pd.to_numeric(frame.iloc[:, 1], errors="coerce"),
            "kz_expected_inflation_12m_pct": pd.to_numeric(frame.iloc[:, 2], errors="coerce"),
            "kz_expected_inflation_5y_pct": pd.to_numeric(frame.iloc[:, 3], errors="coerce"),
        }
    ).dropna(subset=["period_start", "kz_expected_inflation_12m_pct"])
    result["period_end"] = result["period_start"] + pd.offsets.MonthEnd(0)
    # Publication timestamps are not present in the consolidated workbook.
    result["available_date_proxy"] = result["period_end"] + pd.Timedelta(days=31)
    return result.sort_values("period_start")


def normalize_ecb_ciss(contents: dict[str, bytes]) -> pd.DataFrame:
    series = []
    for source_id, content in contents.items():
        frame = pd.read_csv(io.BytesIO(content), usecols=["TIME_PERIOD", "OBS_VALUE"])
        frame = frame.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": source_id})
        frame["date"] = pd.to_datetime(frame["date"])
        frame[source_id] = pd.to_numeric(frame[source_id], errors="coerce")
        series.append(frame.set_index("date"))
    return pd.concat(series, axis=1).reset_index().sort_values("date")


def normalize_cbr_key_rate(content: bytes) -> pd.DataFrame:
    soup = BeautifulSoup(content, "html.parser")
    rows = []
    for tr in soup.select("table.data tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.select("td")]
        if len(cells) != 2 or not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", cells[0]):
            continue
        rows.append({"date": datetime.strptime(cells[0], "%d.%m.%Y").date(), "cbr_key_rate": float(cells[1])})
    return pd.DataFrame(rows).drop_duplicates("date").sort_values("date")


def normalize_nbk_base_rate(contents: dict[int, bytes]) -> pd.DataFrame:
    rows = []
    for year, content in contents.items():
        soup = BeautifulSoup(content, "html.parser")
        for tr in soup.select("table tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.select("td")]
            if len(cells) < 2:
                continue
            raw_date = cells[0].replace("*", "").strip()
            if not re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", raw_date) or not cells[1].strip():
                continue
            rate = float(cells[1].replace(",", "."))
            rows.append(
                {
                    "effective_date": datetime.strptime(raw_date, "%d.%m.%Y").date(),
                    "nbk_base_rate": rate,
                    "source_year": year,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("effective_date").sort_values("effective_date")


def remove_boundary_copy(frame: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Drop an impossible first-date copy of a much later complete FX basket."""
    pivot = frame.pivot_table(
        index="effective_date",
        columns="symbol",
        values="normalized_value",
        aggfunc="last",
    ).sort_index()
    if len(pivot) < 2:
        return frame, []
    first_date = pivot.index[0]
    first = pivot.iloc[0]
    comparable = pivot.drop(index=first_date).dropna(subset=list(first.dropna().index))
    matches = comparable.eq(first, axis="columns").all(axis=1)
    distant_matches = comparable.index[matches & ((comparable.index - first_date).days > 365)]
    if distant_matches.empty:
        return frame, []
    matched_date = distant_matches[-1]
    repaired = frame[~frame["effective_date"].eq(first_date)].copy()
    repair = {
        "source": source_name,
        "action": "dropped_first_effective_date",
        "effective_date": str(pd.Timestamp(first_date).date()),
        "reason": f"complete FX basket duplicates distant date {pd.Timestamp(matched_date).date()}",
    }
    return repaired, [repair]


def validate_units(frame: pd.DataFrame, source_name: str) -> None:
    units_per_symbol = frame.groupby("symbol")["normalized_unit"].nunique()
    if units_per_symbol.gt(1).any():
        symbols = units_per_symbol[units_per_symbol.gt(1)].index.tolist()
        raise ValueError(f"{source_name}: inconsistent normalized units for {symbols}")


def extract_local_official_fx(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    path = repo_root / "data" / "kzt_v0" / "observations.csv"
    frame = pd.read_csv(path, parse_dates=["effective_date", "available_at"])
    observed = frame[(frame["field"] == "close") & frame["is_observation"].eq(1)].copy()
    cbr = observed[observed["source"].eq("CBR")][
        ["effective_date", "available_at", "symbol", "normalized_value", "normalized_unit"]
    ]
    nbk = observed[observed["source"].eq("NBK")][
        ["effective_date", "available_at", "symbol", "normalized_value", "normalized_unit"]
    ]
    validate_units(cbr, "CBR")
    validate_units(nbk, "NBK")
    cbr, cbr_repairs = remove_boundary_copy(cbr, "CBR")
    nbk, nbk_repairs = remove_boundary_copy(nbk, "NBK")
    metadata = {
        "source_id": "project_official_fx_snapshot",
        "input_path": str(path.relative_to(repo_root)),
        "input_sha256": sha256_file(path),
        "warning": "available_at in the branch snapshot is not independently proven; experiment lags values",
        "integrity_repairs": [*cbr_repairs, *nbk_repairs],
    }
    return cbr, nbk, metadata


def describe_frame(frame: pd.DataFrame, date_column: str) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "first_date": str(pd.to_datetime(frame[date_column]).min().date()),
        "last_date": str(pd.to_datetime(frame[date_column]).max().date()),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    retrieval_time = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "generated_at_utc": retrieval_time,
        "pipeline_sha256": sha256_file(Path(__file__).resolve()),
        "date_window": {"from": START_DATE, "to": END_DATE},
        "python": platform.python_version(),
        "libraries": {
            "pandas": pd.__version__,
            "openpyxl": openpyxl.__version__,
            "xlrd": xlrd.__version__,
        },
        "exclusions": {
            "FRED": "not used: current FRED Terms prohibit using content to develop/train ML systems",
            "Bloomberg": "not used: proprietary terminal/feed; public web pages are not a reproducible open API",
        },
        "sources": [],
        "artifacts": {},
    }

    for series in MOEX_SERIES:
        frame, metadata, assembled_raw = fetch_moex(series, args.workers)
        write_bytes(raw_dir / f"{series.series_id}.json", assembled_raw)
        path = normalized_dir / f"{series.series_id}.csv"
        write_frame(path, frame)
        metadata["normalized_path"] = str(path.relative_to(output_dir))
        metadata["normalized_sha256"] = sha256_file(path)
        manifest["sources"].append(metadata)

    fed_content, fed_headers = fetch(FED_H10_URL)
    write_bytes(raw_dir / "fed_h10_daily_indexes.csv", fed_content)
    fed = normalize_fed_h10(fed_content)
    write_frame(normalized_dir / "fed_h10_daily_indexes.csv", fed)
    manifest["sources"].append(
        {
            "source_id": "fed_h10_dollar_indexes",
            "authority": "Board of Governors of the Federal Reserve System",
            "url": FED_H10_URL,
            "license_review": "US federal government source; attribution retained; verify site terms for production ETL",
            "availability": "daily observations released weekly; primary experiment applies an eight-day proxy",
            **describe_frame(fed, "date"),
            "freshness": freshness_metadata(fed, "date", 14),
            "response_headers": fed_headers,
        }
    )

    treasury_contents: dict[int, bytes] = {}
    treasury_headers: dict[int, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, 7)) as pool:
        future_to_year = {pool.submit(fetch, TREASURY_URL.format(year=year)): year for year in range(2020, 2027)}
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            content, headers = future.result()
            treasury_contents[year] = content
            treasury_headers[year] = headers
            write_bytes(raw_dir / f"us_treasury_curve_{year}.xml", content)
    treasury = normalize_treasury(treasury_contents)
    write_frame(normalized_dir / "us_treasury_curve.csv", treasury)
    manifest["sources"].append(
        {
            "source_id": "us_treasury_daily_curve",
            "authority": "U.S. Department of the Treasury",
            "url_template": TREASURY_URL,
            "license_review": "US federal government source; attribution retained",
            "availability": "daily after US market close; experiment applies a conservative lag",
            **describe_frame(treasury, "date"),
            "freshness": freshness_metadata(treasury, "date", 7),
            "response_headers_by_year": treasury_headers,
        }
    )

    eia_content, eia_headers = fetch(EIA_BRENT_URL)
    write_bytes(raw_dir / "eia_brent_daily.xls", eia_content)
    eia = normalize_eia_brent(eia_content)
    write_frame(normalized_dir / "eia_brent_daily.csv", eia)
    manifest["sources"].append(
        {
            "source_id": "eia_brent_daily",
            "authority": "U.S. Energy Information Administration",
            "url": EIA_BRENT_URL,
            "license_review": "EIA public data; cite EIA and retain any source notices",
            "availability": "daily observations updated weekly; primary experiment applies ten calendar days",
            **describe_frame(eia, "date"),
            "freshness": freshness_metadata(eia, "date", 21),
            "response_headers": eia_headers,
        }
    )

    wb_content, wb_headers = fetch(WORLD_BANK_COMMODITIES_URL)
    write_bytes(raw_dir / "world_bank_cmo_historical_monthly.xlsx", wb_content)
    wb = normalize_world_bank(wb_content)
    write_frame(normalized_dir / "world_bank_commodities_monthly.csv", wb)
    manifest["sources"].append(
        {
            "source_id": "world_bank_commodity_prices",
            "authority": "World Bank Prospects Group",
            "url": WORLD_BANK_COMMODITIES_URL,
            "license_review": "CC BY 4.0 according to World Bank dataset catalog",
            "availability": "monthly, second business day; seven-day proxy used; current workbook is not a vintage",
            **describe_frame(wb, "period_start"),
            "freshness": freshness_metadata(wb, "period_end", 45),
            "response_headers": wb_headers,
        }
    )

    kz_cpi_content, kz_cpi_headers = fetch(KZ_CPI_URL)
    write_bytes(raw_dir / "kz_cpi_2022_2026.xlsx", kz_cpi_content)
    kz_cpi = normalize_kz_cpi(kz_cpi_content)
    write_frame(normalized_dir / "kz_cpi_monthly.csv", kz_cpi)
    manifest["sources"].append(
        {
            "source_id": "kz_official_cpi",
            "authority": "Bureau of National Statistics of Kazakhstan",
            "url": KZ_CPI_URL,
            "terms_url": KZ_DATA_USE_URL,
            "license_review": (
                "official statistics may be freely reused, modified and used in software with attribution"
            ),
            "availability": (
                "monthly; experiment uses period-end plus seven calendar days; current workbook is not a vintage"
            ),
            **describe_frame(kz_cpi, "period_start"),
            "freshness": freshness_metadata(kz_cpi, "period_end", 45),
            "response_headers": kz_cpi_headers,
        }
    )

    nbk_form_specs = (
        (
            34,
            "nbk_reserves",
            1_000,
            normalize_nbk_reserves,
            "monthly reserves and National Fund FX assets; 40-day conservative availability proxy",
            "latest_revised_eval_only",
        ),
        (
            479,
            "nbk_current_account",
            5_000,
            normalize_nbk_current_account,
            "quarterly current-account aggregate; 100-day conservative availability proxy",
            "latest_revised_eval_only",
        ),
        (
            35,
            "nbk_kase_monthly_rub",
            1_000,
            normalize_nbk_kase_monthly,
            "monthly KASE RUB volume and end-period average; 40-day conservative availability proxy",
            "short_history_eval_only",
        ),
        (
            181,
            "nbk_interbank_monthly",
            1_000,
            normalize_nbk_interbank_monthly,
            "monthly interbank aggregates; category keys are incomplete in the API response",
            "inventory_only_ambiguous_schema",
        ),
    )
    for form_id, source_id, page_size, normalizer, availability, quality_status in nbk_form_specs:
        query = urllib.parse.urlencode(
            {
                "formId": form_id,
                "page": 0,
                "pageSize": page_size,
                "startDate": START_DATE,
                "endDate": END_DATE,
            }
        )
        url = f"{NBK_OPEN_DATA_URL}?{query}"
        content, headers = fetch(url)
        write_bytes(raw_dir / f"{source_id}.json", content)
        normalized, api_metadata = normalizer(content)
        write_frame(normalized_dir / f"{source_id}.csv", normalized)
        manifest["sources"].append(
            {
                "source_id": source_id,
                "authority": "National Bank of Kazakhstan Open Data",
                "url": url,
                "terms_url": NBK_DATA_TERMS_URL,
                "api_announcement_url": NBK_OPEN_DATA_ANNOUNCEMENT_URL,
                "license_review": (
                    "NBK permits perpetual commercial reuse and modification with source attribution"
                ),
                "availability": availability,
                "data_quality_status": quality_status,
                **api_metadata,
                **describe_frame(normalized, "report_date"),
                "response_headers": headers,
            }
        )

    indicator_query = urllib.parse.urlencode({"from": "2026-02-24", "to": END_DATE})
    nbk_indicators_url = f"{NBK_INDICATORS_URL}?{indicator_query}"
    nbk_indicator_content, nbk_indicator_headers = fetch(nbk_indicators_url)
    write_bytes(raw_dir / "nbk_daily_indicators.json", nbk_indicator_content)
    nbk_indicators = normalize_nbk_indicators(nbk_indicator_content)
    write_frame(normalized_dir / "nbk_daily_indicators.csv", nbk_indicators)
    manifest["sources"].append(
        {
            "source_id": "nbk_daily_indicators",
            "authority": "National Bank of Kazakhstan Open Data",
            "url": nbk_indicators_url,
            "terms_url": NBK_DATA_TERMS_URL,
            "license_review": (
                "NBK permits perpetual commercial reuse and modification with source attribution"
            ),
            "availability": "daily API; history currently begins in 2026 and is not used for model selection",
            "data_quality_status": "prospective_collection_candidate_insufficient_oot_history",
            **describe_frame(nbk_indicators, "date"),
            "freshness": freshness_metadata(nbk_indicators, "date", 7),
            "response_headers": nbk_indicator_headers,
        }
    )

    cbr_reserves_content, cbr_reserves_headers = fetch(CBR_RESERVES_URL)
    write_bytes(raw_dir / "cbr_international_reserves.html", cbr_reserves_content)
    cbr_reserves = normalize_cbr_reserves(cbr_reserves_content)
    write_frame(normalized_dir / "cbr_international_reserves_weekly.csv", cbr_reserves)
    manifest["sources"].append(
        {
            "source_id": "cbr_international_reserves_weekly",
            "authority": "Bank of Russia",
            "url": CBR_RESERVES_URL,
            "license_review": (
                "official public statistics; cite Bank of Russia and confirm automated-use terms for production"
            ),
            "availability": (
                "weekly Friday observation typically released the following Thursday; "
                "experiment applies seven calendar days"
            ),
            "data_quality_status": "operational_candidate_publication_timestamp_not_proven",
            **describe_frame(cbr_reserves, "date"),
            "freshness": freshness_metadata(cbr_reserves, "date", 14),
            "response_headers": cbr_reserves_headers,
        }
    )

    cbr_bci_content, cbr_bci_headers = fetch(CBR_BUSINESS_CLIMATE_URL)
    write_bytes(raw_dir / "cbr_business_climate.json", cbr_bci_content)
    cbr_bci = normalize_cbr_business_climate(cbr_bci_content)
    write_frame(normalized_dir / "cbr_business_climate_monthly.csv", cbr_bci)
    manifest["sources"].append(
        {
            "source_id": "cbr_business_climate_monthly",
            "authority": "Bank of Russia",
            "url": CBR_BUSINESS_CLIMATE_URL,
            "license_review": (
                "official public statistics; cite Bank of Russia and confirm automated-use terms for production"
            ),
            "availability": (
                "monthly; API period marker plus 14 calendar days; current response is not an immutable vintage"
            ),
            "data_quality_status": "latest_revised_eval_only",
            **describe_frame(cbr_bci, "period_end"),
            "freshness": freshness_metadata(cbr_bci, "period_end", 60),
            "response_headers": cbr_bci_headers,
        }
    )

    cbr_current_account_content, cbr_current_account_headers = fetch(CBR_CURRENT_ACCOUNT_URL)
    write_bytes(raw_dir / "cbr_current_account_quarterly.json", cbr_current_account_content)
    cbr_current_account = normalize_cbr_current_account(cbr_current_account_content)
    write_frame(normalized_dir / "cbr_current_account_quarterly.csv", cbr_current_account)
    manifest["sources"].append(
        {
            "source_id": "cbr_current_account_quarterly",
            "authority": "Bank of Russia",
            "url": CBR_CURRENT_ACCOUNT_URL,
            "license_review": (
                "official public statistics; cite Bank of Russia and confirm automated-use terms for production"
            ),
            "availability": (
                "quarterly; period marker plus 90 calendar days; latest API values are revised"
            ),
            "data_quality_status": "latest_revised_eval_only",
            **describe_frame(cbr_current_account, "period_end"),
            "freshness": freshness_metadata(cbr_current_account, "period_end", 250),
            "response_headers": cbr_current_account_headers,
        }
    )

    nbk_bai_content, nbk_bai_headers = fetch(NBK_BUSINESS_ACTIVITY_URL)
    write_bytes(raw_dir / "nbk_business_activity.xlsx", nbk_bai_content)
    nbk_bai = normalize_nbk_business_activity(nbk_bai_content)
    write_frame(normalized_dir / "nbk_business_activity_monthly.csv", nbk_bai)
    manifest["sources"].append(
        {
            "source_id": "nbk_business_activity_monthly",
            "authority": "National Bank of Kazakhstan",
            "url": NBK_BUSINESS_ACTIVITY_URL,
            "terms_url": NBK_DATA_TERMS_URL,
            "license_review": (
                "NBK permits perpetual commercial reuse and modification with source attribution"
            ),
            "availability": (
                "monthly; period-end plus seven calendar days; current workbook is not an immutable vintage"
            ),
            "data_quality_status": "latest_revised_eval_only",
            **describe_frame(nbk_bai, "period_end"),
            "freshness": freshness_metadata(nbk_bai, "period_end", 60),
            "response_headers": nbk_bai_headers,
        }
    )

    nbk_expectations_content, nbk_expectations_headers = fetch(NBK_INFLATION_EXPECTATIONS_URL)
    write_bytes(raw_dir / "nbk_inflation_expectations.xlsx", nbk_expectations_content)
    nbk_expectations = normalize_nbk_inflation_expectations(nbk_expectations_content)
    write_frame(normalized_dir / "nbk_inflation_expectations_monthly.csv", nbk_expectations)
    manifest["sources"].append(
        {
            "source_id": "nbk_inflation_expectations_monthly",
            "authority": "National Bank of Kazakhstan",
            "url": NBK_INFLATION_EXPECTATIONS_URL,
            "terms_url": NBK_DATA_TERMS_URL,
            "license_review": (
                "NBK permits perpetual commercial reuse and modification with source attribution"
            ),
            "availability": (
                "monthly; period-end plus 31 calendar days; current workbook is not an immutable vintage"
            ),
            "data_quality_status": "latest_revised_eval_only",
            **describe_frame(nbk_expectations, "period_end"),
            "freshness": freshness_metadata(nbk_expectations, "period_end", 75),
            "response_headers": nbk_expectations_headers,
        }
    )

    ecb_contents: dict[str, bytes] = {}
    ecb_headers: dict[str, dict[str, str]] = {}
    for source_id, url in ECB_CISS_URLS.items():
        content, headers = fetch(url)
        ecb_contents[source_id] = content
        ecb_headers[source_id] = headers
        write_bytes(raw_dir / f"{source_id}.csv", content)
    ecb = normalize_ecb_ciss(ecb_contents)
    write_frame(normalized_dir / "ecb_ciss_daily.csv", ecb)
    manifest["sources"].append(
        {
            "source_id": "ecb_new_ciss",
            "authority": "European Central Bank",
            "urls": ECB_CISS_URLS,
            "license_review": "ECB permits commercial and non-commercial reuse with attribution",
            "availability": "business-daily, approximately T+1; revisions possible",
            **describe_frame(ecb, "date"),
            "freshness": freshness_metadata(ecb, "date", 14),
            "response_headers_by_series": ecb_headers,
        }
    )

    cbr_content, cbr_headers = fetch(CBR_KEY_RATE_URL)
    write_bytes(raw_dir / "cbr_key_rate.html", cbr_content)
    cbr_rate = normalize_cbr_key_rate(cbr_content)
    write_frame(normalized_dir / "cbr_key_rate.csv", cbr_rate)
    manifest["sources"].append(
        {
            "source_id": "cbr_key_rate",
            "authority": "Bank of Russia",
            "url": CBR_KEY_RATE_URL,
            "license_review": "official public statistics; cite Bank of Russia; verify automated-use terms for production",
            "availability": "daily effective rate; primary experiment waits one calendar day",
            **describe_frame(cbr_rate, "date"),
            "freshness": freshness_metadata(cbr_rate, "date", 7),
            "response_headers": cbr_headers,
        }
    )

    nbk_contents: dict[int, bytes] = {}
    nbk_headers: dict[int, dict[str, str]] = {}
    for year, rubric in NBK_RATE_RUBRICS.items():
        url = f"https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke/rubrics/{rubric}"
        content, headers = fetch(url)
        nbk_contents[year] = content
        nbk_headers[year] = headers
        write_bytes(raw_dir / f"nbk_base_rate_{year}.html", content)
    nbk_rate = normalize_nbk_base_rate(nbk_contents)
    nbk_dates = pd.to_datetime(nbk_rate["effective_date"])
    nbk_year_counts = nbk_dates.dt.year.value_counts().sort_index().to_dict()
    nbk_max_gap_days = int(nbk_dates.sort_values().diff().dt.days.max())
    write_frame(normalized_dir / "nbk_base_rate_events.csv", nbk_rate)
    manifest["sources"].append(
        {
            "source_id": "nbk_base_rate",
            "authority": "National Bank of Kazakhstan",
            "urls": [
                f"https://nationalbank.kz/en/news/grafik-prinyatiya-resheniy-po-bazovoy-stavke/rubrics/{rubric}"
                for rubric in NBK_RATE_RUBRICS.values()
            ],
            "license_review": "official public statistics; cite NBK; production terms/SLA need legal confirmation",
            "availability": "event effective dates; primary experiment waits one calendar day",
            **describe_frame(nbk_rate, "effective_date"),
            "data_quality_status": "excluded_from_models_incomplete_event_history",
            "events_by_year": {str(year): int(count) for year, count in nbk_year_counts.items()},
            "maximum_gap_days": nbk_max_gap_days,
            "response_headers_by_year": nbk_headers,
        }
    )

    cbr_fx, nbk_fx, local_metadata = extract_local_official_fx(args.repo_root.resolve())
    write_frame(normalized_dir / "project_cbr_fx_snapshot.csv", cbr_fx)
    write_frame(normalized_dir / "project_nbk_fx_snapshot.csv", nbk_fx)
    local_metadata.update(
        {
            "cbr_rows": len(cbr_fx),
            "nbk_rows": len(nbk_fx),
            "normalized_paths": ["normalized/project_cbr_fx_snapshot.csv", "normalized/project_nbk_fx_snapshot.csv"],
            "freshness": {
                "cbr": freshness_metadata(cbr_fx, "effective_date", 7),
                "nbk": freshness_metadata(nbk_fx, "effective_date", 7),
            },
        }
    )
    manifest["sources"].append(local_metadata)

    for artifact_dir in (raw_dir, normalized_dir):
        for path in sorted(artifact_dir.iterdir()):
            if path.is_file():
                manifest["artifacts"][str(path.relative_to(output_dir))] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sources": len(manifest["sources"]), "artifacts": len(manifest["artifacts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
