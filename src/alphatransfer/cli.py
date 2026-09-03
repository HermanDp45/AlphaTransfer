"""Small dependency-free command line interface for reproduction."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from .backtest import evaluate_events, walk_forward_backtest, walk_forward_ml_backtest
from .cbr import CURRENCIES, fetch_cbr_daily_history, read_quotes_csv, write_quotes_csv
from .engine import SignalConfig, signal_at, signals_as_of
from .ml import LocalMinimumLogit, ml_signal_at


def _codes(value: str) -> list[str]:
    codes = [x.strip().upper() for x in value.split(",") if x.strip()]
    invalid = set(codes) - set(CURRENCIES)
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported: {', '.join(sorted(invalid))}")
    return codes


def _config(args: argparse.Namespace) -> SignalConfig:
    return SignalConfig(lookback=args.lookback, momentum_days=args.momentum_days, low_percentile=args.low_percentile, rebound_window=args.rebound_window, cooldown_observations=args.cooldown, min_strength=args.min_strength)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphatransfer", description="CBR FX signal layer (all rate values are RUB per one currency unit)")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download normalized CBR history")
    download.add_argument("--codes", type=_codes, default=list(CURRENCIES), help="CSV: TJS,UZS,KGS,AMD,KZT")
    download.add_argument("--start", default=(date.today() - timedelta(days=365 * 6)).isoformat())
    download.add_argument("--end", default=date.today().isoformat())
    download.add_argument("--output", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--lookback", type=int, default=60)
    shared.add_argument("--momentum-days", type=int, default=3)
    shared.add_argument("--low-percentile", type=float, default=0.20)
    shared.add_argument("--rebound-window", type=int, default=20)
    shared.add_argument("--cooldown", type=int, default=3)
    shared.add_argument("--min-strength", type=float, default=0.55)
    signals = sub.add_parser("signals", parents=[shared], help="emit signal table as JSON")
    signals.add_argument("--input", required=True)
    signals.add_argument("--as-of", required=True)
    signals.add_argument("--only-today", action="store_true")
    signals.add_argument("--model", choices=["rules", "classical"], default="rules")
    backtest = sub.add_parser("backtest", parents=[shared], help="evaluate fixed config or walk-forward candidates")
    backtest.add_argument("--input", required=True)
    backtest.add_argument("--as-of", required=True)
    backtest.add_argument("--horizon", type=int, default=5)
    backtest.add_argument("--walk-forward", action="store_true")
    backtest.add_argument("--model", choices=["rules", "classical"], default="rules")
    args = parser.parse_args(argv)
    if args.command == "download":
        quotes = fetch_cbr_daily_history(args.codes, args.start, args.end)
        write_quotes_csv(quotes, args.output)
        print(json.dumps({"output": args.output, "rows": len(quotes), "units": "RUB per 1 recipient-currency unit"}))
        return 0
    quotes = read_quotes_csv(args.input)
    config = _config(args)
    if args.command == "signals":
        if args.model == "classical":
            model = LocalMinimumLogit(horizon=5).fit(quotes, args.as_of, config)
            result = {"model": model.metadata(), "signals": ml_signal_at(quotes, args.as_of, model, config)}
        else:
            result = signal_at(quotes, args.as_of, config) if args.only_today else signals_as_of(quotes, args.as_of, config)
    elif args.model == "classical":
        # Classical ML is always evaluated by folds: a single in-sample score
        # would contradict the point-in-time requirement.
        result = walk_forward_ml_backtest(quotes, config, horizon=args.horizon)
    elif args.walk_forward:
        candidates = [config, SignalConfig(lookback=90, momentum_days=3, low_percentile=.15, rebound_window=30, cooldown_observations=3, min_strength=.55), SignalConfig(lookback=40, momentum_days=2, low_percentile=.25, rebound_window=15, cooldown_observations=4, min_strength=.55)]
        result = walk_forward_backtest(quotes, candidates, horizon=args.horizon)
    else:
        cutoff = date.fromisoformat(args.as_of)
        result = evaluate_events(quotes, signals_as_of(quotes, cutoff, config), horizon=args.horizon, evaluation_end=cutoff)
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
