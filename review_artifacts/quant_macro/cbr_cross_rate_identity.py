#!/usr/bin/env python3
"""Audit compatibility with the post-2024 CBR cross-rate mechanism.

The historical KZT check uses only the branch snapshot.  The five-corridor
snapshot is a transparent arithmetic reproduction of primary-source values
verified on 2026-09-04; it is a mechanism demo, not a backtest feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_START = pd.Timestamp("2024-12-27")
CBR_METHOD_URL = "https://www.cbr.ru/Queries/XsltBlock/File/105012/-1/2531"
CBR_PAIR_LIST_URL = "https://www.cbr.ru/Content/Document/File/186003/Annex_OD-1012_en.pdf"
CBR_DAILY_URL = "https://www.cbr.ru/eng/currency_base/daily/"

LIVE_USD_RUB = 86.8872
LIVE_EFFECTIVE_DATE = "2026-09-04"
LIVE_LOCAL_VALUES = (
    ("AMD", 100, 364.19, 23.8577, "Central Bank of Armenia", "https://www.cba.am/en/exchange-rates-archive"),
    ("KGS", 100, 87.4488, 99.3578, "National Bank of the Kyrgyz Republic", "https://www.nbkr.kg/getservice.jsp?lang=ENG"),
    ("KZT", 100, 455.40, 19.0793, "National Bank of Kazakhstan", "https://nationalbank.kz/rss/get_rates.cfm?fdate=03.09.2026"),
    ("TJS", 10, 9.2525, 93.9067, "National Bank of Tajikistan", "https://www.nbt.tj/en/kurs/kurs.php"),
    ("UZS", 10_000, 11_813.09, 73.5516, "Central Bank of Uzbekistan", "https://cbu.uz/en/arkhiv-kursov-valyut/"),
)


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[2])
    parser.add_argument("--output-dir", type=Path, default=script.parent / "mechanism_audit")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_series(observations: pd.DataFrame, source: str, symbol: str, name: str) -> pd.DataFrame:
    columns = ["effective_date", "normalized_value"]
    if source == "CBR" and symbol == "KZT":
        columns.extend(["raw_nominal", "raw_value"])
    result = observations[
        observations["source"].eq(source)
        & observations["symbol"].eq(symbol)
        & observations["field"].eq("close")
        & observations["is_observation"].eq(1)
    ][columns].copy()
    return (
        result.drop_duplicates("effective_date")
        .sort_values("effective_date")
        .rename(columns={"normalized_value": name})
    )


def historical_kzt_identity(observations: pd.DataFrame) -> pd.DataFrame:
    cbr_kzt = official_series(observations, "CBR", "KZT", "actual_rub_per_kzt")
    cbr_usd = official_series(observations, "CBR", "USD", "cbr_rub_per_usd")
    nbk_usd = official_series(observations, "NBK", "USD", "nbk_kzt_per_usd").rename(
        columns={"effective_date": "nbk_rate_date"}
    )
    cbr = cbr_kzt.merge(cbr_usd, on="effective_date", validate="one_to_one")
    # В snapshot нет publication timestamp. Strict-prior effective-date
    # alignment выбран как отдельная арифметическая sensitivity и не
    # идентифицирует фактический denominator, использованный ЦБ.
    joined = pd.merge_asof(
        cbr.sort_values("effective_date"),
        nbk_usd.sort_values("nbk_rate_date"),
        left_on="effective_date",
        right_on="nbk_rate_date",
        direction="backward",
        allow_exact_matches=False,
    )
    joined = joined[joined["effective_date"].ge(METHOD_START)].dropna(
        subset=["nbk_kzt_per_usd"]
    )
    joined["synthetic_rub_per_kzt"] = (
        joined["cbr_rub_per_usd"] / joined["nbk_kzt_per_usd"]
    )
    joined["synthetic_cbr_value"] = (
        joined["synthetic_rub_per_kzt"] * joined["raw_nominal"]
    ).round(4)
    joined["exact_to_cbr_rounding"] = joined["synthetic_cbr_value"].eq(
        joined["raw_value"].round(4)
    )
    joined["relative_error_bps"] = 10_000.0 * (
        joined["synthetic_rub_per_kzt"] / joined["actual_rub_per_kzt"] - 1.0
    )
    return joined.sort_values("effective_date")


def live_five_corridor_identity() -> pd.DataFrame:
    rows = []
    for corridor, nominal, local_per_usd, actual, authority, source_url in LIVE_LOCAL_VALUES:
        synthetic = round(LIVE_USD_RUB * nominal / local_per_usd, 4)
        rows.append(
            {
                "cbr_effective_date": LIVE_EFFECTIVE_DATE,
                "corridor": corridor,
                "nominal": nominal,
                "cbr_rub_per_usd": LIVE_USD_RUB,
                "local_currency_per_usd": local_per_usd,
                "synthetic_cbr_value_per_nominal": synthetic,
                "actual_cbr_value_per_nominal": actual,
                "exact_to_four_decimals": synthetic == actual,
                "local_authority": authority,
                "local_source_url": source_url,
                "cbr_source_url": CBR_DAILY_URL,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = repo_root / "data" / "kzt_v0" / "observations.csv"
    observations = pd.read_csv(
        observations_path,
        parse_dates=["effective_date"],
    )
    historical = historical_kzt_identity(observations)
    live = live_five_corridor_identity()
    if historical.empty or not live["exact_to_four_decimals"].all():
        raise ValueError("cross-rate compatibility audit failed")
    historical_path = output_dir / "historical_kzt_identity.csv"
    live_path = output_dir / "live_five_corridor_identity_2026-09-04.csv"
    historical.to_csv(historical_path, index=False, lineterminator="\n")
    live.to_csv(live_path, index=False, lineterminator="\n")
    summary = {
        "status": "mechanism_compatibility_audit_not_predictive_backtest",
        "cbr_method_start": str(METHOD_START.date()),
        "historical_kzt": {
            "rows": len(historical),
            "exact_rounding_matches": int(historical["exact_to_cbr_rounding"].sum()),
            "exact_rounding_match_rate": float(historical["exact_to_cbr_rounding"].mean()),
            "mean_absolute_relative_error_bps": float(
                historical["relative_error_bps"].abs().mean()
            ),
            "maximum_absolute_relative_error_bps": float(
                historical["relative_error_bps"].abs().max()
            ),
        },
        "live_five_corridor": {
            "rows": len(live),
            "exact_rounding_matches": int(live["exact_to_four_decimals"].sum()),
            "exact_rounding_match_rate": float(live["exact_to_four_decimals"].mean()),
        },
        "interpretation": (
            "The five contemporaneous CBR corridor targets are arithmetically compatible with "
            "one RUB anchor and national-bank USD/local denominators. This is not forecast alpha. "
            "A NOW-h model must forecast the future path of both cross components and the Alpha "
            "executable-quote residual, conditional on what is known at decision_ts."
        ),
        "point_in_time_warning": (
            "The live table is a manually transcribed one-day arithmetic illustration, without an "
            "independent raw-response/hash lineage for all five local inputs. Historical KZT uses "
            "a selected strict-prior effective-date alignment from the branch snapshot; publication "
            "timestamps are absent, so it neither identifies the actual CBR denominator nor proves PIT."
        ),
        "alignment": "selected_strict_prior_nbk_effective_date_not_publication_time",
        "legacy_output_filename_note": (
            "Historical output filenames retain the word identity for compatibility; the supported "
            "claim is arithmetic compatibility on the selected alignment."
        ),
        "sources": {
            "cbr_method": CBR_METHOD_URL,
            "cbr_pair_list": CBR_PAIR_LIST_URL,
            "cbr_daily": CBR_DAILY_URL,
        },
        "hashes": {
            str(observations_path.relative_to(repo_root)): sha256(observations_path),
            str(Path(__file__).resolve().relative_to(repo_root)): sha256(Path(__file__).resolve()),
            historical_path.name: sha256(historical_path),
            live_path.name: sha256(live_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
