"""Official-source ingestion with raw cache, retries and strict schemas."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .normalization import cbr_rub_per_unit, moex_rub_per_kzt, nbk_rub_per_kzt
from .schema import Observation


CBR_IDS = {"KZT": "R01335", "USD": "R01235", "CNY": "R01375"}
UTC = timezone.utc


def _context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore[import-not-found]
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class CachedHttp:
    def __init__(self, root: str | Path, retries: int = 3, timeout: int = 40):
        self.root, self.retries, self.timeout = Path(root), retries, timeout
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, *, suffix: str) -> bytes:
        digest = hashlib.sha256(url.encode()).hexdigest()
        path = self.root / f"{digest}.{suffix}"
        url_path = self.root / f"{digest}.url"
        if path.exists():
            if not url_path.exists():
                url_path.write_text(url, encoding="utf-8")
            return path.read_bytes()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AlphaTransfer-v0/0.2 contact=research"})
                with urllib.request.urlopen(request, timeout=self.timeout, context=_context()) as response:
                    payload = response.read()
                if not payload:
                    raise ValueError("empty HTTP response")
                path.write_bytes(payload)
                url_path.write_text(url, encoding="utf-8")
                return payload
            except Exception as exc:  # network boundary; retain final cause
                last_error = exc
                if attempt + 1 < self.retries:
                    import time as clock
                    clock.sleep(0.4 * (2 ** attempt))
        raise RuntimeError(f"source fetch failed after {self.retries} attempts: {url}") from last_error


def _at_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def _at_close(day: date) -> datetime:
    return datetime.combine(day, time(20, 0), tzinfo=UTC)


def fetch_cbr(client: CachedHttp, codes: list[str], start: date, end: date) -> list[Observation]:
    rows: list[Observation] = []
    for code in codes:
        if code not in CBR_IDS:
            raise ValueError(f"unsupported CBR code: {code}")
        params = urllib.parse.urlencode({
            "date_req1": start.strftime("%d/%m/%Y"), "date_req2": end.strftime("%d/%m/%Y"),
            "VAL_NM_RQ": CBR_IDS[code],
        })
        payload = client.get(f"https://www.cbr.ru/scripts/XML_dynamic.asp?{params}", suffix="xml")
        root = ET.fromstring(payload)
        if root.tag != "ValCurs":
            raise ValueError(f"CBR schema changed: root={root.tag!r}")
        for node in root.findall("Record"):
            if node.find("Nominal") is None or node.find("Value") is None or "Date" not in node.attrib:
                raise ValueError("CBR schema changed: Record lacks Date/Nominal/Value")
            day = datetime.strptime(node.attrib["Date"], "%d.%m.%Y").date()
            nominal = float(node.findtext("Nominal", "0"))
            value = float(node.findtext("Value", "0").replace(",", "."))
            normalized = cbr_rub_per_unit(value, nominal)
            unit = f"RUB_per_{code}"
            rows.append(Observation(day, _at_start(day), "CBR", code, "close", nominal, value, normalized, unit))
    if not rows:
        raise ValueError("CBR returned no observations")
    return rows


def _parse_nbk_day(payload: bytes, expected_day: date, codes: set[str]) -> list[Observation]:
    root = ET.fromstring(payload)
    if root.tag != "rates" or root.find("date") is None:
        raise ValueError(f"NBK schema changed: root={root.tag!r} or date missing")
    effective = datetime.strptime(root.findtext("date", ""), "%d.%m.%Y").date()
    if effective != expected_day:
        raise ValueError(f"NBK returned {effective} for requested {expected_day}")
    # The official service explicitly represents unavailable archive dates as
    # <info>...нет...</info>. This is a data gap, not a schema failure.
    if root.find("info") is not None and not root.findall("item"):
        return []
    found: dict[str, Observation] = {}
    for item in root.findall("item"):
        code = (item.findtext("title") or "").strip().upper()
        if code not in codes:
            continue
        if item.find("description") is None or item.find("quant") is None:
            raise ValueError("NBK schema changed: item lacks description/quant")
        value = float(item.findtext("description", "0").replace(",", "."))
        nominal = float(item.findtext("quant", "0"))
        normalized = nbk_rub_per_kzt(value, nominal) if code == "RUB" else value / nominal
        unit = "RUB_per_KZT" if code == "RUB" else f"KZT_per_{code}"
        found[code] = Observation(effective, _at_start(effective), "NBK", code, "close", nominal, value, normalized, unit)
    missing = codes - found.keys()
    if missing:
        raise ValueError(f"NBK schema/content changed: missing {sorted(missing)} on {expected_day}")
    return list(found.values())


def fetch_nbk(client: CachedHttp, codes: list[str], start: date, end: date, workers: int = 12) -> list[Observation]:
    wanted = set(codes)
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)

    def one(day: date) -> list[Observation]:
        url = "https://nationalbank.kz/rss/get_rates.cfm?" + urllib.parse.urlencode({"fdate": day.strftime("%d.%m.%Y")})
        return _parse_nbk_day(client.get(url, suffix="xml"), day, wanted)

    rows: list[Observation] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, day): day for day in days}
        for future in as_completed(futures):
            rows.extend(future.result())
    return sorted(rows, key=lambda r: (r.effective_date, r.symbol))


def _description_facevalue(client: CachedHttp, security: str) -> float:
    url = f"https://iss.moex.com/iss/securities/{security}.json?iss.meta=off&iss.only=description"
    body = json.loads(client.get(url, suffix="json"))
    try:
        columns = body["description"]["columns"]
        records = body["description"]["data"]
        name_i, value_i = columns.index("name"), columns.index("value")
        return float(next(r[value_i] for r in records if r[name_i] == "FACEVALUE"))
    except (KeyError, ValueError, StopIteration, TypeError) as exc:
        raise ValueError("MOEX description schema changed; FACEVALUE unavailable") from exc


def fetch_moex(client: CachedHttp, security: str, start: date, end: date, expected_facevalue: float) -> list[Observation]:
    facevalue = _description_facevalue(client, security)
    if abs(facevalue - expected_facevalue) > 1e-9:
        raise ValueError(f"MOEX FACEVALUE changed: expected {expected_facevalue}, got {facevalue}")
    rows: list[Observation] = []
    offset = 0
    required = ["open", "close", "high", "low", "begin", "end"]
    while True:
        params = urllib.parse.urlencode({"interval": 24, "from": start.isoformat(), "till": end.isoformat(), "start": offset, "iss.meta": "off", "iss.only": "candles"})
        url = f"https://iss.moex.com/iss/engines/currency/markets/selt/securities/{security}/candles.json?{params}"
        body = json.loads(client.get(url, suffix="json"))
        if "candles" not in body or "columns" not in body["candles"] or "data" not in body["candles"]:
            raise ValueError("MOEX candle schema changed")
        columns, data = body["candles"]["columns"], body["candles"]["data"]
        if any(name not in columns for name in required):
            raise ValueError(f"MOEX candle schema changed: required {required}, got {columns}")
        if not data:
            break
        indexes = {name: columns.index(name) for name in required}
        for record in data:
            day = datetime.fromisoformat(record[indexes["begin"]]).date()
            for field in ("open", "high", "low", "close"):
                raw = record[indexes[field]]
                if raw is None:
                    raise ValueError(f"MOEX null {field} at {day}")
                rows.append(Observation(day, _at_close(day), "MOEX", security, field, facevalue, float(raw), moex_rub_per_kzt(float(raw), facevalue), "RUB_per_KZT"))
        offset += len(data)
    if not rows:
        raise ValueError("MOEX returned no candles")
    return rows


def fetch_all(raw_cache: str | Path, start: date, end: date, source_cfg: dict) -> list[Observation]:
    client = CachedHttp(raw_cache)
    return (
        fetch_cbr(client, list(source_cfg["cbr_codes"]), start, end)
        + fetch_nbk(client, list(source_cfg["nbk_codes"]), start, end)
        + fetch_moex(client, str(source_cfg["moex_security"]), start, end, float(source_cfg["moex_facevalue"]))
    )
