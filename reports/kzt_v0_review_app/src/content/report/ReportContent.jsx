import React from "react";

import {
  DataComponent, DataTable, MetricCard, ReportSection, RichNarrative, useDataApp,
} from "../../data-app-public.jsx";

const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

export function ReportContent() {
  const { snapshot, visible, canEdit, mode, appTitle, setAppTitle } = useDataApp();
  const primary = snapshot.queries.primary_metrics.rows;
  const robustness = snapshot.queries.source_robustness.rows;
  const quality = snapshot.queries.quality.rows;
  const current = snapshot.queries.current_signal.rows;
  const favorable = primary.find((row) => row.scenario === "favorable_now");
  const closing = primary.find((row) => row.scenario === "window_closing");
  const signal = current[0];

  return <article className="report-content" aria-label="RUB to KZT V0 report">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>
        {appTitle}
      </h1>
      <RichNarrative id="report:description" className="report-deck" label="Edit report introduction"
        value="Target — нормализованный курс ЦБ РФ. НБК и MOEX не смешиваются в синтетическую истину: они проверяют переносимость результата. Итоговый исследовательский gate — **no-go**." />
    </header>

    {visible("report-summary") && <ReportSection id="report-summary" title="Почему no-go"
      queryId="primary_metrics" queryIds={["primary_metrics", "source_robustness"]}
      sourceRowsByQuery={{ primary_metrics: primary, source_robustness: robustness }} showHeading={false}>
      <RichNarrative id="report-summary:body" className="report-summary-lead" label="Edit main finding"
        value={`## Почему no-go\n\n**favorable_now** дал ${favorable.signals} out-of-time сигналов: hit rate ${pct(favorable.hitRate)} против ${pct(favorable.randomHitRate)} у случайного допустимого дня, lift ${favorable.lift}. Но на тех же датах lift НБК и MOEX равен 0. **window_closing** формально достиг lift ${closing.lift}, однако его random hit rate уже ${pct(closing.randomHitRate)}. Основания недостаточно устойчивы для запуска отправки.`} />
    </ReportSection>}

    <div className="report-facts" aria-label="Key results">
      <MetricCard id="report-metric-lift" title="CBR target lift" queryId="primary_metrics" sourceRows={primary}
        value={String(favorable.lift)} description={`${favorable.signals} favorable_now signals at h=5`} />
      <MetricCard id="report-metric-signal" title="Signal on 2026-09-03" queryId="current_signal" sourceRows={current}
        value={signal.scenario ?? "None"} description={`eligible_to_send=${signal.eligibleToSend}; ${signal.suppressedReason}`} />
    </div>

    <section className="report-section">
      <RichNarrative id="report-primary:lead" value="## Out-of-time метрики\n\n36 месяцев train → 6 месяцев validation → 3 месяца untouched test, шаг 3 месяца. Порог и гиперпараметры выбираются только на предыдущем validation; false positive стоит 3× false negative." />
      <DataComponent id="report-primary-metrics" title="Primary h=5 scenarios" queryId="primary_metrics"
        kind="table" sourceRows={primary} displayRows={primary} description="Random baseline uses the same untouched periods and MOEX-eligible dates.">
        <DataTable rows={primary} searchable={false} rowKey="scenario" caption="Primary horizon metrics"
          columns={[
            { key: "scenario", label: "Scenario" }, { key: "signals", label: "Signals" },
            { key: "hitRate", label: "Hit rate", presentation: "percent" },
            { key: "randomHitRate", label: "Random", presentation: "percent" },
            { key: "lift", label: "Lift" }, { key: "timingBps", label: "Timing, bps" },
            { key: "brier", label: "Brier" },
          ]} />
      </DataComponent>
    </section>

    <section className="report-section">
      <RichNarrative id="report-robustness:lead" value="## Robustness по источникам\n\nТот же поток favorable_now проверен против локальных минимумов каждого нормализованного ряда. CBR — заданный target; нулевой перенос на НБК и MOEX блокирует общий go." />
      <DataComponent id="report-robustness" title="Same candidate dates, independent truth" queryId="source_robustness"
        kind="table" sourceRows={robustness} displayRows={robustness}>
        <DataTable rows={robustness} searchable={false} rowKey="source" caption="Source robustness"
          columns={[{ key: "source", label: "Source" }, { key: "signals", label: "Signals" },
            { key: "hitRate", label: "Hit rate", presentation: "percent" },
            { key: "randomHitRate", label: "Random", presentation: "percent" }, { key: "lift", label: "Lift" }]} />
      </DataComponent>
    </section>

    <ReportSection id="report-quality" title="Data and method limits" queryId="quality" sourceRows={quality} showHeading={false}>
      <RichNarrative id="report-quality:body" label="Edit limitations"
        value={`## Data and method limits\n\nModel grain: one row per fresh MOEX candle; ${quality[0].modelRows} rows from ${quality[0].commonStart} to ${quality[0].end}. Full calendar retains ${quality[0].calendarDays} days, including ${quality[0].closedDays} days without a MOEX candle. Earlier NBK archive gaps shorten the common interval. Historical available_at is reconstructed conservatively; statutory holiday flags never create a signal. V0 does not validate product conversion, deep links, or stale-price UX.`} />
    </ReportSection>
  </article>;
}
