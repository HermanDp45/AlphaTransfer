"""Command line orchestration for fetch, backtest, signal and report."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

from .config import V0Config
from .evaluation import fit_live, walk_forward
from .features import build_calendar_rows, build_feature_rows, read_calendar, read_features, write_calendar, write_features
from .ingestion import fetch_all
from .policy import apply_policy, public_signal, raw_candidates
from .reporting import build_snapshot, write_report
from .schema import read_observations, write_observations


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def run_fetch(cfg: V0Config, start: date, end: date) -> dict:
    rows = fetch_all(cfg.path("raw_cache"), start, end, cfg.section("sources"))
    write_observations(rows, cfg.path("observations"))
    return {"observations": len(rows), "start": start.isoformat(), "end": end.isoformat(), "output": str(cfg.path("observations"))}


def run_backtest(cfg: V0Config) -> dict:
    observations = read_observations(cfg.path("observations"))
    write_calendar(build_calendar_rows(observations), cfg.path("calendar"))
    rows = build_feature_rows(observations, cfg.section("features"))
    write_features(rows, cfg.path("features"))
    result = walk_forward(rows, cfg.raw)
    _json(cfg.path("backtest"), result)
    return result


def run_signal(cfg: V0Config, as_of: date) -> dict:
    rows = read_features(cfg.path("features"))
    backtest = json.loads(cfg.path("backtest").read_text(encoding="utf-8"))
    eligible_rows = [r for r in rows if date.fromisoformat(r["date"]) <= as_of]
    if not eligible_rows:
        raise ValueError("as_of precedes available MOEX history")
    session_day = date.fromisoformat(eligible_rows[-1]["date"])
    model, threshold = fit_live(rows, session_day, backtest, cfg.raw)
    probabilities = [model.predict(r) for r in eligible_rows]
    policy_cfg = cfg.section("policy")
    rebound = float(backtest["last_selection"]["rebound_threshold_bps"])
    candidates = raw_candidates(eligible_rows, probabilities, threshold, rebound, int(policy_cfg["window_closing_lookback"]))
    selected = apply_policy(candidates, int(policy_cfg["cooldown_sessions"]), int(policy_cfg["max_candidates_7d"]), backtest.get("primary_lifts"))
    today = [x for x in selected if x["date"] == session_day.isoformat()]
    row = eligible_rows[-1]
    if today:
        candidate = today[-1]
        lift = float(backtest["primary_lifts"].get(candidate["scenario"], 0.0))
        result = public_signal(candidate, row, model.explain(row), lift, float(cfg.raw["model"]["min_oot_lift"]))
    else:
        result = {"as_of": session_day.isoformat(), "corridor": "RUB_KZT", "scenario": None, "confidence": 0.0,
                  "eligible_to_send": False,
                  "rate_snapshot": {k: round(float(row[k]), 8) for k in ("cbr_rate", "nbk_rate", "moex_close", "usd_cross_rate", "cny_cross_rate")},
                  "source_freshness": {}, "evidence": [], "suppressed_reason": "no_candidate"}
    src = cfg.section("sources")
    result["source_freshness"] = {
        "cbr": {"age_days": int(row["cbr_age_days"]), "is_fresh": int(row["cbr_age_days"]) <= int(src["cbr_fresh_days"])},
        "nbk": {"age_days": int(row["nbk_age_days"]), "is_fresh": int(row["nbk_age_days"]) <= int(src["nbk_fresh_days"])},
        "moex": {"age_days": (as_of - session_day).days, "is_fresh": as_of == session_day},
    }
    if as_of != session_day:
        age = (as_of - session_day).days
        result["as_of"] = as_of.isoformat(); result["eligible_to_send"] = False
        result["stale_signal_as_of"] = session_day.isoformat()
        result["suppressed_reason"] = "market_closed_signal_stale" if age <= int(src["signal_ttl_calendar_days"]) else "market_closed_ttl_expired"
    _json(cfg.path("signal"), result)
    return result


def run_report(cfg: V0Config) -> dict:
    rows = read_features(cfg.path("features"))
    backtest = json.loads(cfg.path("backtest").read_text(encoding="utf-8"))
    signal = json.loads(cfg.path("signal").read_text(encoding="utf-8"))
    calendar = read_calendar(cfg.path("calendar"))
    snapshot = build_snapshot(rows, backtest, signal, calendar)
    _json(cfg.path("report_snapshot"), snapshot)
    html_path, dist = write_report(snapshot, rows, cfg.path("report_html"), cfg.path("report_app"), calendar)
    return {"decision": snapshot["decision"], "html": str(html_path), "dist": str(dist)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alphatransfer", description="Transparent RUB→KZT V0")
    parser.add_argument("--config", default="configs/kzt_v0.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download and normalize CBR, NBK and MOEX")
    fetch.add_argument("--start"); fetch.add_argument("--end", default=date.today().isoformat())
    sub.add_parser("backtest", help="build point-in-time features and expanding walk-forward")
    signal = sub.add_parser("signal", help="produce public signal JSON for an arbitrary date")
    signal.add_argument("--as-of", default=date.today().isoformat())
    sub.add_parser("report", help="build deterministic HTML report and dist/index.html")
    all_ = sub.add_parser("all", help="fetch → backtest → signal → report")
    all_.add_argument("--start"); all_.add_argument("--end", default=date.today().isoformat()); all_.add_argument("--as-of")
    args = parser.parse_args(argv); cfg = V0Config.load(args.config)
    if args.command in ("fetch", "all"):
        end = date.fromisoformat(args.end)
        default_start = date(end.year - int(cfg.raw["project"]["history_years"]), end.month, min(end.day, 28))
        fetch_result = run_fetch(cfg, date.fromisoformat(args.start) if args.start else default_start, end)
        if args.command == "fetch": print(json.dumps(fetch_result, ensure_ascii=False, indent=2)); return 0
    if args.command in ("backtest", "all"):
        backtest_result = run_backtest(cfg)
        if args.command == "backtest": print(json.dumps({"folds": len(backtest_result["folds"]), "primary_lifts": backtest_result["primary_lifts"], "output": str(cfg.path("backtest"))}, ensure_ascii=False, indent=2)); return 0
    if args.command in ("signal", "all"):
        as_of = date.fromisoformat(args.as_of or args.end) if args.command == "all" else date.fromisoformat(args.as_of)
        signal_result = run_signal(cfg, as_of)
        if args.command == "signal": print(json.dumps(signal_result, ensure_ascii=False, indent=2)); return 0
    report_result = run_report(cfg); print(json.dumps(report_result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
