"""CBR daily-rate download and nominal normalisation.

CBR publishes a number of roubles for a *nominal* amount of currency.  This
module always exposes ``rub_per_unit``: roubles required for one unit of the
recipient currency.  Lower is therefore better for a client converting RUB.
"""

from __future__ import annotations

import csv
import io
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


def _ssl_context() -> ssl.SSLContext:
    """Use certifi when a locally installed Python lacks system CAs.

    This keeps TLS verification enabled.  Some macOS framework Pythons do not
    expose the operating-system trust store to ``urllib`` even though the
    machine itself can reach CBR successfully.
    """
    try:
        import certifi  # type: ignore[import-not-found]
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class Currency:
    code: str
    cbr_id: str
    country: str


# CBR Valute IDs are stable identifiers used by XML_dynamic.asp.
CURRENCIES: dict[str, Currency] = {
    "AMD": Currency("AMD", "R01060", "Armenia"),
    "KGS": Currency("KGS", "R01370", "Kyrgyzstan"),
    "KZT": Currency("KZT", "R01335", "Kazakhstan"),
    "TJS": Currency("TJS", "R01670", "Tajikistan"),
    "UZS": Currency("UZS", "R01717", "Uzbekistan"),
}


@dataclass(frozen=True)
class Quote:
    date: date
    corridor: str
    rub_per_unit: float
    nominal: int
    cbr_value: float


def _d(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _cbr_date(value: date) -> str:
    return value.strftime("%d/%m/%Y")


def fetch_cbr_daily_history(
    codes: Iterable[str], start: date | str, end: date | str, *, timeout: int = 30
) -> list[Quote]:
    """Download CBR publication days for requested ISO codes.

    It deliberately does not fill weekends: those would create fake zero
    movements.  Use :func:`calendar_daily` only for display/reporting.
    """
    start_date, end_date = _d(start), _d(end)
    if start_date > end_date:
        raise ValueError("start must be no later than end")
    output: list[Quote] = []
    for code in codes:
        code = code.upper()
        if code not in CURRENCIES:
            raise ValueError(f"Unsupported corridor {code!r}; use one of {sorted(CURRENCIES)}")
        params = urllib.parse.urlencode(
            {
                "date_req1": _cbr_date(start_date),
                "date_req2": _cbr_date(end_date),
                "VAL_NM_RQ": CURRENCIES[code].cbr_id,
            }
        )
        request = urllib.request.Request(
            f"https://www.cbr.ru/scripts/XML_dynamic.asp?{params}",
            headers={"User-Agent": "AlphaTransfer-reproducible-research/0.1"},
        )
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            payload = response.read()
        root = ET.fromstring(payload)
        for node in root.findall("Record"):
            raw_date = node.attrib["Date"]
            quote_date = datetime.strptime(raw_date, "%d.%m.%Y").date()
            nominal = int((node.findtext("Nominal") or "").strip())
            # CBR uses a Russian decimal comma in XML.
            value = float((node.findtext("Value") or "").strip().replace(",", "."))
            if nominal <= 0:
                raise ValueError(f"Invalid nominal {nominal} for {code} at {quote_date}")
            output.append(Quote(quote_date, code, value / nominal, nominal, value))
    return sorted(output, key=lambda q: (q.corridor, q.date))


def write_quotes_csv(quotes: Iterable[Quote], path: str | Path) -> None:
    """Write an inspectable, locale-independent normalized quote file."""
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "corridor", "rub_per_unit", "nominal", "cbr_value"])
        writer.writeheader()
        for q in quotes:
            writer.writerow({"date": q.date.isoformat(), "corridor": q.corridor, "rub_per_unit": f"{q.rub_per_unit:.12g}", "nominal": q.nominal, "cbr_value": f"{q.cbr_value:.12g}"})


def read_quotes_csv(path: str | Path) -> list[Quote]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "corridor", "rub_per_unit"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must include columns {sorted(required)}")
        return sorted(
            [
                Quote(
                    date.fromisoformat(row["date"]), row["corridor"].upper(), float(row["rub_per_unit"]),
                    int(row.get("nominal") or 1), float(row.get("cbr_value") or row["rub_per_unit"]),
                )
                for row in reader
            ],
            key=lambda q: (q.corridor, q.date),
        )
