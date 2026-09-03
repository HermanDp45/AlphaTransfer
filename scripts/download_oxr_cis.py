#!/usr/bin/env python3
"""Download daily Open Exchange Rates history and derive RUB-based CIS rates."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


API_ROOT = "https://openexchangerates.org/api"
EARLIEST_DATE = date(1999, 1, 1)
QUOTES = ("AMD", "AZN", "BYN", "KGS", "KZT", "MDL", "TJS", "TMT", "UZS")
DEFAULT_DB = Path("data/open_exchange_rates/rub_cis.sqlite3")
DEFAULT_CSV = Path("data/open_exchange_rates/rub_cis_daily.csv")


@dataclass(frozen=True)
class Usage:
    plan: str
    requests_used: int
    requests_remaining: int
    requests_quota: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the newest possible contiguous OXR daily history and derive "
            "CIS currency rates with RUB as the base."
        )
    )
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--max-new-days",
        type=int,
        help="Limit new paid requests; by default, use all remaining monthly requests.",
    )
    parser.add_argument(
        "--usage-only",
        action="store_true",
        help="Show plan and quota without consuming a request.",
    )
    return parser.parse_args()


def get_app_id() -> str:
    app_id = os.environ.get("OXR_APP_ID")
    if app_id:
        return app_id
    return getpass.getpass("Open Exchange Rates App ID: ").strip()


def api_json(path: str, app_id: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    query = {"app_id": app_id}
    if params:
        query.update(params)
    url = f"{API_ROOT}/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "AlphaTransfer-research/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace").replace(app_id, "***")
        raise RuntimeError(f"OXR HTTP {error.code}: {body[:300]}") from None
    except urllib.error.URLError as error:
        reason = str(error.reason).replace(app_id, "***")
        raise RuntimeError(f"OXR connection error: {reason[:300]}") from None


def get_usage(app_id: str) -> Usage:
    payload = api_json("usage.json", app_id)
    data = payload["data"]
    plan = data["plan"]
    usage = data["usage"]
    return Usage(
        plan=str(plan["name"]),
        requests_used=int(usage["requests"]),
        requests_remaining=int(usage["requests_remaining"]),
        requests_quota=int(usage["requests_quota"]),
    )


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rates (
            rate_date TEXT NOT NULL,
            pair TEXT NOT NULL,
            base TEXT NOT NULL,
            quote TEXT NOT NULL,
            quote_per_rub REAL NOT NULL,
            rub_per_quote REAL NOT NULL,
            rub_per_usd REAL NOT NULL,
            quote_per_usd REAL NOT NULL,
            published_at_utc TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (rate_date, quote)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_usage (
            utc_month TEXT PRIMARY KEY,
            requests_reserved INTEGER NOT NULL
        )
        """
    )
    return connection


def existing_complete_dates(connection: sqlite3.Connection) -> set[date]:
    rows = connection.execute(
        "SELECT rate_date FROM rates GROUP BY rate_date HAVING COUNT(*) = ?", (len(QUOTES),)
    )
    return {date.fromisoformat(row[0]) for row in rows}


def contiguous_days(end: date, existing: set[date]) -> int:
    count = 0
    current = end
    while current in existing:
        count += 1
        current -= timedelta(days=1)
    return count


def local_requests_reserved(connection: sqlite3.Connection, utc_month: str) -> int:
    row = connection.execute(
        "SELECT requests_reserved FROM monthly_usage WHERE utc_month = ?", (utc_month,)
    ).fetchone()
    return int(row[0]) if row else 0


def reserve_requests(connection: sqlite3.Connection, utc_month: str, count: int) -> None:
    connection.execute(
        """
        INSERT INTO monthly_usage (utc_month, requests_reserved)
        VALUES (?, ?)
        ON CONFLICT (utc_month) DO UPDATE
        SET requests_reserved = requests_reserved + excluded.requests_reserved
        """,
        (utc_month, count),
    )
    connection.commit()


def newest_target_dates(end: date, total_days: int) -> list[date]:
    available_days = (end - EARLIEST_DATE).days + 1
    count = min(total_days, available_days)
    return [end - timedelta(days=offset) for offset in range(count)]


def fetch_day(day: date, app_id: str) -> tuple[date, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return day, api_json(f"historical/{day.isoformat()}.json", app_id)
        except RuntimeError as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def rows_from_payload(day: date, payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    if payload.get("base") != "USD":
        raise ValueError(f"{day}: expected USD source base, got {payload.get('base')!r}")

    rates = payload["rates"]
    missing = sorted({"RUB", *QUOTES} - rates.keys())
    if missing:
        raise ValueError(f"{day}: missing currencies: {', '.join(missing)}")

    rub_per_usd = float(rates["RUB"])
    published_at = datetime.fromtimestamp(int(payload["timestamp"]), tz=timezone.utc).isoformat()
    result = []
    for quote in QUOTES:
        quote_per_usd = float(rates[quote])
        quote_per_rub = quote_per_usd / rub_per_usd
        result.append(
            (
                day.isoformat(),
                f"RUB/{quote}",
                "RUB",
                quote,
                quote_per_rub,
                1.0 / quote_per_rub,
                rub_per_usd,
                quote_per_usd,
                published_at,
                "openexchangerates.org",
            )
        )
    return result


def save_day(connection: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT OR REPLACE INTO rates (
            rate_date, pair, base, quote, quote_per_rub, rub_per_quote,
            rub_per_usd, quote_per_usd, published_at_utc, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def export_csv(connection: sqlite3.Connection, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    columns = (
        "date",
        "pair",
        "base",
        "quote",
        "quote_per_rub",
        "rub_per_quote",
        "rub_per_usd",
        "quote_per_usd",
        "published_at_utc",
        "source",
    )
    rows = connection.execute(
        """
        SELECT rate_date, pair, base, quote, quote_per_rub, rub_per_quote,
               rub_per_usd, quote_per_usd, published_at_utc, source
        FROM rates
        ORDER BY rate_date, quote
        """
    )
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    current_utc_date = datetime.now(timezone.utc).date()
    if args.end >= current_utc_date:
        print("--end must be the last completed UTC day, not today or a future date", file=sys.stderr)
        return 2
    if args.end < EARLIEST_DATE:
        print(f"--end must not be earlier than {EARLIEST_DATE}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("--workers must be positive", file=sys.stderr)
        return 2

    app_id = get_app_id()
    if not app_id:
        print("App ID is empty", file=sys.stderr)
        return 2

    usage = get_usage(app_id)
    print(
        f"Plan: {usage.plan}; usage: {usage.requests_used}/{usage.requests_quota}; "
        f"remaining: {usage.requests_remaining}"
    )
    if args.usage_only:
        return 0

    connection = connect_db(args.db)
    utc_month = current_utc_date.strftime("%Y-%m")
    local_reserved = local_requests_reserved(connection, utc_month)
    if usage.requests_quota < 0:
        allowance = (args.end - EARLIEST_DATE).days + 1
    else:
        allowance = usage.requests_remaining
        allowance = min(allowance, max(usage.requests_quota - local_reserved, 0))
    if args.max_new_days is not None:
        allowance = min(allowance, max(args.max_new_days, 0))
    if usage.requests_remaining <= 0:
        print("No historical-request quota remains; exporting already downloaded data.")
    elif allowance <= 0:
        print("New downloads disabled by --max-new-days; exporting already downloaded data.")

    existing = existing_complete_dates(connection)
    already_contiguous = contiguous_days(args.end, existing)
    targets = newest_target_dates(args.end, already_contiguous + allowance)
    missing = [day for day in targets if day not in existing]

    if targets:
        print(
            f"Newest contiguous range target: {targets[-1]}..{targets[0]} "
            f"({len(targets)} days); new requests: {len(missing)}"
        )
    else:
        print("No dates are available to export yet.")
    if local_reserved:
        print(f"Locally reserved this UTC month: {local_reserved} requests")
    reserve_requests(connection, utc_month, len(missing))

    failures: list[tuple[date, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_day, day, app_id): day for day in missing}
        for future in as_completed(futures):
            day = futures[future]
            try:
                _, payload = future.result()
                save_day(connection, rows_from_payload(day, payload))
                completed += 1
                if completed % 25 == 0 or completed == len(missing):
                    print(f"Downloaded {completed}/{len(missing)} new days")
            except Exception as error:  # noqa: BLE001 - every failed date must be checkpointed
                message = str(error).replace(app_id, "***")[:300]
                failures.append((day, message))
                print(f"Failed {day}: {message}", file=sys.stderr)

    export_csv(connection, args.csv)
    complete = existing_complete_dates(connection)
    final_contiguous = contiguous_days(args.end, complete)
    final_start = args.end - timedelta(days=max(final_contiguous - 1, 0))
    connection.close()
    print(f"CSV: {args.csv}")
    print(f"Contiguous coverage: {final_start}..{args.end} ({final_contiguous} days)")
    print(f"Pairs: {', '.join(f'RUB/{quote}' for quote in QUOTES)}")
    if failures:
        print(f"Failed dates: {len(failures)}; rerun the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
