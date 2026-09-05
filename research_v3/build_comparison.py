#!/usr/bin/env python3
"""Build the consolidated retrospective ledger from finished local artifacts.

No training, network access or edits to upstream reports. All policy values use
the corridor candidate layer. Delta Brier always references the OLD CNY-basis
method on the same horizon and track; matched short-history ICE is kept apart.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, os, time
from datetime import datetime, timezone
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
HORIZONS=(1,3,5,10,20)
OLD_MODEL="hgb_plus_cnyrub_basis"
TRACKS={"development":"development_2023_2025", "development_2023_2025":"development_2023_2025", "diagnostic_2026":"diagnostic_2026"}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(row,*columns,default=math.nan):
    for c in columns:
        if c in row and pd.notna(row[c]):return float(row[c])
    return default

def flags_for(model,track,origin):
    flags=["retrospective_exploratory","synthetic_behavior_excluded"]
    if track=="diagnostic_2026":flags.append("partial_2026_already_inspected")
    if "__fred" in model or "__cboe" in model or "__all_new" in model:flags.append("restricted_source_counterfactual")
    if "__treasury" in model:flags.append("direct_Treasury_FRED_replica")
    if "__gpr" in model or "__all_new" in model:flags.append("GPR_latest_snapshot_not_vintage")
    if "stale" in model:flags.append("lag_sensitivity")
    if "long_combo" in origin:flags+= ["posthoc_long_macro_combination","macro_missing_before_2020"]
    if ("120m" in model or "allm" in model or "long" in model) and "official" not in model:flags.append("long_train_partial_exogenous_history")
    if model.startswith("monthly_"):flags.append("monthly_refit_policy_history")
    return flags

def make_record(row,track,horizon,path,layout):
    model=str(row["config_id"])
    track=TRACKS[track]
    origin=str(path.parent.relative_to(ROOT))
    if layout=="models":
        brier=number(row,"brier");lift=number(row,"lift")
        count=number(row,"signals");forward=number(row,"forward_delta_bps")
        hit=number(row,"hit_rate");rows=number(row,"rows");dates=number(row,"dates")
        coverage=number(row,"mean_cell_week_coverage")
    else:
        brier=number(row,"brier","model_brier")
        lift=number(row,"candidate_cell_standardized_lift")
        count=number(row,"candidate_signal_count")
        forward=number(row,"candidate_cell_standardized_forward_bps_delta")
        hit=number(row,"candidate_hit_rate");rows=number(row,"eligible_count","prediction_rows")
        dates=rows/5 if math.isfinite(rows) else math.nan
        coverage=number(row,"mean_candidate_weeks_with_1_to_2_signals_share")
    return dict(model=model,track=track,horizon=int(horizon),brier=brier,lift=lift,
                candidate_count=int(count) if math.isfinite(count) else None,
                candidate_hit_rate=hit,forward_delta_bps=forward,
                mean_candidate_weekly_coverage=coverage,prediction_rows=int(rows) if math.isfinite(rows) else None,
                decision_dates=int(dates) if math.isfinite(dates) else None,
                origin_folder=origin,source_file=str(path.relative_to(ROOT)),
                comparison_scope="old_method_same_horizon_and_track",synthetic_behavior_excluded=True,
                flags=flags_for(model,track,origin))

def stable_inputs(wait_seconds):
    """Wait at most 60 s for upstream completion, then read a stable snapshot."""
    deadline=time.monotonic()+min(max(wait_seconds,0),60)
    required=[HERE/"models/parity.json",HERE/"external_data/verification.json",HERE/"tabm/output/_SUCCESS.json",HERE/"tabm/output/incumbent_parity.json"]
    required += [HERE/"models"/f"summary_h{h}.csv" for h in HORIZONS]
    while True:
        try:
            assert all(p.exists() for p in required),"required upstream output is absent"
            parity=json.loads(required[0].read_text());assert parity["status"]=="PASS"
            assert parity["maximum_probability_error"]<1e-12 and parity["candidate_mask_exact"]
            ext=json.loads(required[1].read_text());assert ext["status"]=="passed"
            success=json.loads(required[2].read_text());assert success["status"]=="complete"
            tabm=json.loads(required[3].read_text());assert all(r["status"]=="PASS" for r in tabm)
            for filename in ("aggregate_metrics.csv","scorecard.csv"):
                assert sha(HERE/"tabm/output"/filename)==success["output_hashes"][filename],filename+" hash mismatch"
            return dict(root_parity=parity,external_verification=ext,tabm_parity=tabm)
        except (AssertionError,KeyError,ValueError,json.JSONDecodeError) as exc:
            if time.monotonic()>=deadline:raise RuntimeError("Upstream artifacts not complete/consistent: "+str(exc)) from exc
            time.sleep(min(2,max(0,deadline-time.monotonic())))

def build():
    paths=[];records=[];baseline={};checks=[]
    def read(path):
        paths.append(path)
        return pd.read_csv(path)
    # Canonical frozen incumbent for all development horizons and diagnostic h5.
    for track,filename in [("development_2023_2025","development_metrics.csv"),("diagnostic_2026","diagnostic_2026_metrics.csv")]:
        path=ROOT/"final_solution/model_bundle"/filename
        d=read(path)
        for _,r in d[d.config_id.eq(OLD_MODEL)].iterrows():
            record=make_record(r,track,int(r.horizon_cbr_rows_pub_proxy),path,"core")
            record["flags"].append("canonical_old_frozen_baseline")
            record["is_canonical_baseline"]=True
            record["baseline_status"]="frozen_v2"
            baseline[(track,record["horizon"])]=record
            records.append(record)
    # All root model/window/objective experiments.
    for h in HORIZONS:
        path=HERE/"models"/f"summary_h{h}.csv"
        d=read(path)
        assert not d.duplicated(["config_id","track"]).any(),str(path)
        for _,r in d.iterrows():
            record=make_record(r,str(r.track),h,path,"models")
            if record["model"]=="baseline_reproduction":
                key=(record["track"],h)
                if key in baseline:
                    old=baseline[key]
                    for field in ("brier","lift","forward_delta_bps"):
                        assert abs(record[field]-old[field])<1e-10,(key,field,record[field],old[field])
                    assert record["candidate_count"]==old["candidate_count"]
                    checks.append(dict(check="old_baseline_summary_parity",track=key[0],horizon=h,status="PASS"))
                else:
                    assert record["track"]=="diagnostic_2026",key
                    record["is_canonical_baseline"]=True
                    record["baseline_status"]="old_method_recomputed_no_frozen_diagnostic"
                    record["flags"].append("old_method_recomputed_no_frozen_diagnostic")
                    baseline[key]=record
            records.append(record)
    for sub in ("", "long_combo/"):
        for track in ("development","diagnostic_2026"):
            names=[track+"_metrics.csv"]
            if not sub:names.append(track+"_stale_metrics.csv")
            for name in names:
                path=HERE/"external_data"/sub/name
                for _,r in read(path).iterrows():
                    records.append(make_record(r,track,int(r.horizon_cbr_rows_pub_proxy),path,"core"))
    # ICE changes the training support and is explicitly isolated from the main delta.
    path=HERE/"external_data/credit_matched_2026_metrics.csv"
    ice=read(path)
    icebase=ice[ice.config_id.eq("credit_matched_2026__basis")].iloc[0]
    for _,r in ice.iterrows():
        record=make_record(r,"diagnostic_2026",5,path,"core")
        record["comparison_scope"]="separate_ICE_train2024_validation2025"
        record["matched_baseline_model"]="credit_matched_2026__basis"
        record["matched_baseline_brier"]=float(icebase.brier)
        record["matched_delta_brier"]=record["brier"]-float(icebase.brier)
        record["matched_relative_brier_improvement"]=1-record["brier"]/float(icebase.brier)
        record["flags"]+= ["short_ICE_history_separate_matched_comparison","no_delta_to_main_old_baseline"]
        records.append(record)
    # Prefer full aggregate: scorecard is cross-checked, never used as a second row set.
    path=HERE/"tabm/output/aggregate_metrics.csv"
    tabm=read(path)
    card=read(HERE/"tabm/output/scorecard.csv")
    merged=tabm.merge(card,on=["period","config_id"],suffixes=("_aggregate","_scorecard"),validate="one_to_one")
    assert len(merged)==len(tabm)==len(card)
    assert (merged.model_brier_aggregate-merged.model_brier_scorecard).abs().max()<1e-12
    checks.append(dict(check="TabM_scorecard_aggregate_parity",rows=len(merged),status="PASS"))
    for _,r in tabm.iterrows():
        records.append(make_record(r,str(r.period),int(r.horizon_cbr_rows_pub_proxy),path,"core"))
    # Fill only the old same-horizon denominator. Never use a model's own ablation
    # comparator as if it were the incumbent; ICE deliberately stays NaN here.
    for r in records:
        r.setdefault("is_canonical_baseline",False)
        if r["comparison_scope"].startswith("separate_ICE"):
            r["baseline_status"]="not_comparable_to_main_training_protocol"
        else:
            b=baseline[(r["track"],r["horizon"])]
            assert r["prediction_rows"]==b["prediction_rows"],(r["model"],"different target support")
            r.update(baseline_model=OLD_MODEL,baseline_brier=b["brier"],baseline_lift=b["lift"],baseline_candidate_count=b["candidate_count"],baseline_forward_delta_bps=b["forward_delta_bps"],baseline_source_file=b["source_file"],baseline_status=b["baseline_status"],delta_brier=r["brier"]-b["brier"],relative_brier_improvement=1-r["brier"]/b["brier"],lift_delta=r["lift"]-b["lift"],candidate_count_delta=r["candidate_count"]-b["candidate_count"],forward_delta_vs_old_bps=r["forward_delta_bps"]-b["forward_delta_bps"])
            if b["baseline_status"]!="frozen_v2":r["flags"].append(b["baseline_status"])
    # Total-vs-original and incremental-vs-long are different estimands. Keep
    # their intervals and denominators in different columns, with source hashes.
    total_path=HERE/"models/external_combo_vs_incumbent_ci.csv"
    total_ci=read(total_path)
    for _,ci in total_ci.iterrows():
        matching=[r for r in records if r["model"]==ci.config_id and r["track"]==ci.track and r["horizon"]==5 and r["origin_folder"].endswith("external_data/long_combo")]
        assert len(matching)==1,(ci.config_id,ci.track)
        r=matching[0]
        assert abs(r["delta_brier"]-float(ci.delta_brier))<1e-12
        r.update(delta_brier_ci95_low=float(ci.ci95_low),delta_brier_ci95_high=float(ci.ci95_high),delta_brier_ci_source=str(total_path.relative_to(ROOT)),delta_brier_ci_status="exploratory_post_selection_unadjusted",delta_brier_improved_years=int(ci.improved_years),delta_brier_improved_cells=int(ci.improved_cells))
    for track,filename in [("development_2023_2025","development_paired_audit.csv"),("diagnostic_2026","diagnostic_2026_paired_audit.csv")]:
        path=HERE/"external_data/long_combo"/filename
        for _,ci in read(path).iterrows():
            matches=[r for r in records if r["model"]==ci.candidate and r["track"]==track and r["origin_folder"].endswith("external_data/long_combo")]
            assert len(matches)==1
            matches[0].update(matched_baseline_model=str(ci.baseline),matched_delta_brier=float(ci.brier_delta),matched_relative_brier_improvement=float(ci.relative_brier_improvement),matched_delta_brier_ci95_low=float(ci.ci95_low),matched_delta_brier_ci95_high=float(ci.ci95_high),matched_ci_source=str(path.relative_to(ROOT)))
    for r in records:
        r["flags"]=";".join(dict.fromkeys(r["flags"]))
    ledger=pd.DataFrame(records)
    cols=["model","track","horizon","brier","delta_brier","relative_brier_improvement","delta_brier_ci95_low","delta_brier_ci95_high","delta_brier_ci_status","delta_brier_ci_source","delta_brier_improved_years","delta_brier_improved_cells","lift","lift_delta","candidate_count","candidate_count_delta","candidate_hit_rate","forward_delta_bps","forward_delta_vs_old_bps","mean_candidate_weekly_coverage","prediction_rows","decision_dates","baseline_model","baseline_brier","baseline_lift","baseline_candidate_count","baseline_forward_delta_bps","baseline_status","baseline_source_file","comparison_scope","is_canonical_baseline","matched_baseline_model","matched_baseline_brier","matched_delta_brier","matched_relative_brier_improvement","matched_delta_brier_ci95_low","matched_delta_brier_ci95_high","matched_ci_source","origin_folder","source_file","synthetic_behavior_excluded","flags"]
    ledger=ledger.reindex(columns=cols).sort_values(["horizon","track","origin_folder","model"]).reset_index(drop=True)
    assert not ledger.duplicated(["model","track","horizon","source_file"]).any()
    assert not ledger.model.str.contains("synthetic|behavior",case=False).any()
    return ledger,paths,checks

def markdown(ledger):
    # Fixed category representatives, not a programmatic selection of whichever
    # model has the best looked-at score. All experiments remain in the CSV.
    wanted=[
        ("final_solution/model_bundle",OLD_MODEL,"Старый CNY basis"),
        ("research_v3/models","basis_train_120m","Train 120 месяцев"),
        ("research_v3/models","annual_recent_calibration_3m","Calibration 3 месяца"),
        ("research_v3/models","long_with_historical_cny","Long + ранняя история CNY"),
        ("research_v3/models","long_short_ensemble","Long/short ensemble"),
        ("research_v3/tabm/output","tabm_plus_cnyrub_basis","TabM + basis"),
        ("research_v3/tabm/output","blend_plus_cnyrub_basis","HGB/TabM blend"),
        ("research_v3/external_data","hgb_plus_cnyrub_basis__treasury_stale","Treasury inflation lag7"),
        ("research_v3/external_data","hgb_plus_cnyrub_basis__all_new","FRED + CBOE + GPR"),
        ("research_v3/external_data/long_combo","basis_train_120m__treasury_lag7","Long120 + Treasury lag7"),
    ]
    def table(track):
        lines=["| Вариант | Brier ↓ | Улучшение Brier к старому | Lift | Кандидаты | Reference gain, bps | Δ gain к старому, bps |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for origin,model,label in wanted:
            m=ledger[(ledger.horizon==5)&(ledger.track==track)&(ledger.origin_folder==origin)&(ledger.model==model)]
            assert len(m)==1,(track,model,len(m))
            r=m.iloc[0]
            lines.append(f"| {label} | {r.brier:.6f} | {100*r.relative_brier_improvement:+.2f}% | {r.lift:.3f} | {int(r.candidate_count)} | {r.forward_delta_bps:.2f} | {r.forward_delta_vs_old_bps:+.2f} |")
        return "\n".join(lines)
    return f"""# Сводное сравнение AlphaTransfer v3

Реестр содержит **{len(ledger)} строк** по моделям, горизонтам и периодам: [COMPARISON.csv](COMPARISON.csv). Он пересобирается командой `python research_v3/build_comparison.py` из готовых исходных таблиц. Повторяющиеся модели из разных веток сохранены с `origin_folder` и `source_file`, чтобы provenance не потерялся.

**Все основные Δ Brier рассчитаны к старому `hgb_plus_cnyrub_basis` на том же горизонте и периоде.** Положительное относительное улучшение означает уменьшение Brier; `delta_brier = new − old`, поэтому для него отрицательное значение лучше. Lift, число кандидатов и reference gain везде относятся к `candidate_signal`, а не к синтетическому портфелю или доставленным пушам.

## h=5: development 2023–2025

{table('development_2023_2025')}

Здесь 3 635 строк, сгруппированных в 727 дат. Reference gain — средняя разница с базой внутри год×коридор для официального курса, в bps. Это не исполненная банковская экономия. Улучшение Brier не означает улучшения policy-выгоды: например, long120 + Treasury улучшает proper score и одновременно снижает lift/reference gain. Нельзя превращать выбор лучшей просмотренной строки в подтверждение на независимой выборке.

**Два разных сравнения для long120 + Treasury:** общий эффект **к старому incumbent** — Δ Brier **−0.007322**, 95% month-block CI **[−0.014351; −0.000610]**, улучшение в 2/3 годах и 12/15 ячейках; этот интервал **не включает ноль**. Добавочный эффект **к уже длинной модели** — Δ **−0.001556**, CI **[−0.004148; +0.000898]**, улучшение в 3/3 годах и 13/15 ячейках; этот интервал включает ноль. Первый результат не доказывает, что именно Treasury дал весь прирост. Оба интервала retrospective/post-selection, без коррекции всей просмотренной семьи моделей, поэтому не являются confirmatory. В CSV общий эффект и интервал находятся в `delta_brier*`, добавочный — в `matched_*`; baseline не подменяется. Источники: [общий эффект](models/external_combo_vs_incumbent_ci.csv), [добавочный эффект](external_data/long_combo/development_paired_audit.csv).

## h=5: diagnostic 2026

{table('diagnostic_2026')}

Период неполный и ранее просмотрен: 780 строк, 156 дат. Он не используется как новый holdout. Для long120 + Treasury общий Δ Brier к старому равен **+0.002311**, CI **[−0.008411; +0.014146]**: point estimate ухудшается, интервал включает ноль.

## Границы сопоставимости

- Для development h=1/3/5/10/20 и diagnostic h=5 denominator взят прямо из frozen v2 `final_solution/model_bundle`. Корневые `baseline_reproduction` проверены против этих значений по Brier, lift, candidate count и reference gain.
- Для diagnostic h=1/3/10/20 frozen v2 таблица отсутствует. Использовано повторное исполнение **старого метода** `baseline_reproduction` на том же горизонте; строки помечены `old_method_recomputed_no_frozen_diagnostic`. Горизонты между собой не смешиваются.
- ICE имеет отдельный train2024/validation2025 protocol. Его глобальные `delta_brier` и `relative_brier_improvement` намеренно пусты; сравнение с matched baseline лежит в `matched_*`. Эту небольшую исследовательскую выборку нельзя представить как обычный old-baseline uplift.
- Synthetic behavior simulation исключена из FX/ML-ledger; результаты симуляции не смешиваются с фактическими прогнозами курсов. Флаг `synthetic_behavior_excluded=True` отражает это для каждой строки.
- FRED/CBOE/all-new отмечены как restricted-source counterfactual. Treasury-реплика, GPR latest-snapshot caveat, лаговая sensitivity и post-hoc long combo отражены в `flags`. Отсутствие numerical Bloomberg ablation не заменено выдуманной строкой; доступность источников описана в [source_access_decisions.csv](external_data/source_access_decisions.csv).
- Пропуски внешних признаков в ранней длинной истории, лимиты публикационного времени, отсутствие executable quotes и многократный просмотр данных сохраняются. Сводный CSV не повышает статус моделей до production или подтверждённой клиентской выгоды.

Источники: `models/summary_h*.csv`; `external_data/*metrics.csv`; `external_data/long_combo/*metrics.csv`; `tabm/output/aggregate_metrics.csv`, сверенный со `scorecard.csv`. [COMPARISON_MANIFEST.json](COMPARISON_MANIFEST.json) фиксирует hashes, finished parity и проверки.
"""

def write_atomic(path,content):
    temporary=path.with_name("."+path.name+".tmp")
    temporary.write_text(content,encoding="utf-8")
    os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--wait-seconds",type=int,default=60)
    args=parser.parse_args();parity=stable_inputs(args.wait_seconds)
    # Re-read once if any source file changed while the matrix was being read.
    ledger,paths,checks=build();before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    ledger2,paths2,checks2=build();after={str(p.relative_to(ROOT)):sha(p) for p in paths2}
    assert before==after,"Upstream changed during aggregation; rerun after completion"
    ledger=ledger2;checks=checks2
    write_atomic(HERE/"COMPARISON.csv",ledger.to_csv(index=False,float_format="%.15g"))
    write_atomic(HERE/"FINAL_COMPARISON.md",markdown(ledger))
    manifest=dict(status="complete",generated_at_utc=datetime.now(timezone.utc).isoformat(),rows=len(ledger),tracks=sorted(ledger.track.unique()),horizons=sorted(ledger.horizon.unique().tolist()),metric_layer="corridor_candidate",synthetic_behavior_excluded=True,baseline_model=OLD_MODEL,checks=checks,upstream_finished_parity=parity,input_sha256=after,script_sha256=sha(Path(__file__)),output_sha256={name:sha(HERE/name) for name in ("COMPARISON.csv","FINAL_COMPARISON.md")})
    write_atomic(HERE/"COMPARISON_MANIFEST.json",json.dumps(manifest,indent=2))
    print(json.dumps({"status":"complete","rows":len(ledger),"checks":len(checks),"files":["COMPARISON.csv","FINAL_COMPARISON.md","COMPARISON_MANIFEST.json"]}))

if __name__=="__main__":main()
