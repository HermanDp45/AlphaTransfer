import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Tooltip } from "recharts";
import { chartPointerTooltipPosition } from "../dashboard-ask.js";

import { categoryLabel, displayValue, label as humanize, tick } from "./chart-theme.js";
import { scatterTooltipIdentityField } from "./chart-data-shape.js";
import { orderTooltipEntries } from "./chart-transforms.js";

function MeasuredTooltipContent({ content, measure, ...props }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    const card = ref.current?.firstElementChild;
    measure(card);
    if (!card) return;
    const observer = new ResizeObserver(() => measure(card));
    observer.observe(card);
    return () => { observer.disconnect(); measure(null); };
  }, [measure, props.active, props.payload]);
  return <div ref={ref} style={{ display: "contents" }}>
    {React.isValidElement(content) ? React.cloneElement(content, props)
      : typeof content === "function" ? content(props) : content}
  </div>;
}

/** Item tooltips follow the pointer, not the center of a cell or pie sector.
 * Position updates stay in this leaf; the chart/rows don't rerender on every move.
 * Keyboard navigation retains Recharts' datum anchor; selected cards stay pinned.
 */
export function PointerChartTooltip({ frameRef, content, ...props }) {
  const pointer = useRef(null), card = useRef(null), frame = useRef(null);
  const [position, setPosition] = useState(undefined);
  const place = useCallback(() => {
    const root = frameRef.current;
    if (!root || !card.current || !pointer.current || root.querySelector("[data-chart-tooltip-pinned]")) return;
    const chart = root.querySelector(".recharts-wrapper");
    if (!chart) return;
    const bounds = chart.getBoundingClientRect();
    const cardBounds = card.current.getBoundingClientRect();
    if (!bounds.width || !bounds.height || !cardBounds.width || !cardBounds.height) return;
    const next = chartPointerTooltipPosition(bounds, { width: chart.offsetWidth, height: chart.offsetHeight },
      cardBounds, { width: window.innerWidth, height: window.innerHeight }, pointer.current);
    setPosition(current => current?.x === next.x && current?.y === next.y ? current : next);
  }, [frameRef]);
  const measure = useCallback(element => { card.current = element; if (element) place(); }, [place]);
  useEffect(() => {
    const root = frameRef.current;
    if (!root) return;
    const schedule = () => { if (frame.current == null) frame.current = requestAnimationFrame(() => { frame.current = null; place(); }); };
    const move = event => { if (event.pointerType === "touch") return; pointer.current = { x: event.clientX, y: event.clientY }; schedule(); };
    const reset = () => { pointer.current = null; setPosition(undefined); };
    const focus = event => { if (event.target.matches?.(":focus-visible")) reset(); };
    root.addEventListener("pointermove", move);
    root.addEventListener("pointerover", move);
    root.addEventListener("pointerleave", reset);
    root.addEventListener("focusin", focus);
    window.addEventListener("resize", schedule);
    document.addEventListener("scroll", schedule, true);
    return () => {
      cancelAnimationFrame(frame.current); frame.current = null;
      root.removeEventListener("pointermove", move); root.removeEventListener("pointerover", move);
      root.removeEventListener("pointerleave", reset);
      root.removeEventListener("focusin", focus); window.removeEventListener("resize", schedule);
      document.removeEventListener("scroll", schedule, true);
    };
  }, [frameRef, place]);
  return <Tooltip {...props} position={position} allowEscapeViewBox={{ x: true, y: true }}
    content={<MeasuredTooltipContent content={content} measure={measure} />} />;
}

export function ChartTooltip({
  active,
  label,
  payload = [],
  stacked = false,
  vertical = false,
  resolveColor,
  mode = "default",
  xField,
  yField,
  groupField,
  xLabel,
  yLabel,
  formatValue,
  details,
  detailFields = [],
  headerValue,
  children,
  formatLabel, resolveStyle, comparisonMode = false, baseField = field => field,
}) {
  if (active && details) return <div className={`chart-tooltip chart-tooltip--plain${headerValue != null ? " chart-tooltip--details" : ""}`}>
    {headerValue != null ? <div className="chart-tooltip-heading"><strong>{label}</strong><b>{headerValue}</b></div>
      : label != null && <strong>{label}</strong>}
    {details.map(({ label: name, value }) => <span key={name}>{name}<b>{value}</b></span>)}
    {children}
  </div>;
  if (!active || !payload.length) return null;
  const row = payload[0]?.payload;
  if (mode === "pie") label = row?.[xField] ?? label;
  const ageUnit = /^(weeks?|days?|months?)\b/i.exec(xLabel ?? "")?.[1];
  if (ageUnit && label !== "" && label != null && Number.isFinite(Number(label))) {
    label = `${ageUnit.replace(/s$/i, "").replace(/^./, letter => letter.toUpperCase())} ${label}`;
  }
  if (mode === "heatmap" && row) {
    return (
      <div className="chart-tooltip chart-tooltip--plain">
        <strong>{String(tick(row[xField] ?? ""))}</strong>
        <span>
          {detailFields.find(({ field }) => field === groupField)?.label ?? humanize(String(groupField))}
          <b>{categoryLabel(groupField, row[groupField] ?? "—")}</b>
        </span>
        <span>
          {yLabel ?? humanize(String(yField))}
          <b>{row.__unknown ? "Not yet observed" : formatValue ? formatValue(row[yField], yField) : displayValue(row[yField])}</b>
        </span>
        {detailFields.filter(({ field }) => field !== groupField).map(({ field, label }) => <span key={field}>{label}<b>{displayValue(row[field])}</b></span>)}
      </div>
    );
  }
  if (mode === "scatter" && row) {
    const identityField = scatterTooltipIdentityField(row);
    const identity = identityField ? row[identityField] : undefined;
    return (
      <div className="chart-tooltip chart-tooltip--plain">
        {identity && <strong>{String(identity)}</strong>}
        <span>
          {xLabel ?? humanize(String(xField))}
          <b>{formatValue ? formatValue(row[xField], xField) : displayValue(row[xField])}</b>
        </span>
        <span>
          {yLabel ?? humanize(String(yField))}
          <b>{formatValue ? formatValue(row[yField], yField) : displayValue(row[yField])}</b>
        </span>
      </div>
    );
  }
  if (mode === "boxPlot" && row) {
    const statistics = [
      ["Maximum", row.maximum],
      ["75th percentile", row.upperQuartile],
      ["Median", row.median],
      ["25th percentile", row.lowerQuartile],
      ["Minimum", row.minimum],
    ];
    return (
      <div className="chart-tooltip chart-tooltip--plain chart-tooltip--distribution">
        <strong>{String(tick(row[xField] ?? label ?? ""))}</strong>
        {statistics.map(([name, value]) => (
          <span key={name} data-box-statistic={name}>
            {name}
            <b>{formatValue ? formatValue(value, yField) : displayValue(value)}</b>
          </span>
        ))}
      </div>
    );
  }
  const seenFields = new Set();
  const items = payload
    .filter((item) => {
      if (item.value == null || item.dataKey === "baseline") return false;
      const field = String(item.dataKey ?? item.name);
      if (seenFields.has(field)) return false;
      seenFields.add(field);
      return true;
    })
    .map((item) => {
      if (item.dataKey !== "magnitude" && item.dataKey !== "range") {
        return resolveColor ? { ...item, color: resolveColor(item) ?? item.color } : item;
      }
      const isTotal = Boolean(item.payload?.isTotal);
      const change = Number(item.payload?.change ?? item.value);
      const color = change < 0 ? "var(--negative)" : "var(--positive)";
      return {
        ...item,
        name: "Net change",
        value: isTotal ? item.payload?.runningTotal ?? item.payload?.balance : change,
        color: isTotal ? "var(--chart-neutral-fill, color-mix(in srgb, var(--text) 3%, var(--surface)))" : color,
        ...(isTotal
          ? {
              name: item.payload.totalType === "beginning" ? "Beginning total" : "Ending total",
            }
          : {}),
      };
    });
  const ordered = orderTooltipEntries(items, { stacked, vertical });
  // Comparison order must not flip when the previous value exceeds the current one.
  if (ordered.some(item => /previous/i.test(item.dataKey))) ordered.sort((a,b) => Number(/previous/i.test(a.dataKey)) - Number(/previous/i.test(b.dataKey)));
  const comparisonKeys = [...new Set(ordered.map(item => baseField(item.dataKey)))];
  if (comparisonMode && comparisonKeys.length > 1 && ordered.some(item => /previous/i.test(item.dataKey))) {
    const previous = ordered.filter(item => /previous/i.test(item.dataKey));
    const current = comparisonKeys.map(key => ordered.find(item => item.dataKey === key)
      ?? { ...ordered.find(item => baseField(item.dataKey) === key), dataKey:key, name:key, value:null });
    return <div className="chart-tooltip chart-tooltip-comparison">
      <div className="comparison-tooltip-grid"><div className="comparison-tooltip-row comparison-tooltip-heading"><span/><span data-current-period>{label != null ? categoryLabel(xField,tick(label)) : "Current"}</span><span>{formatLabel?.(previous[0]) ?? "Previous"}</span></div>
      {current.map(item => { const prior = previous.find(prior => baseField(prior.dataKey) === item.dataKey);
        return <div key={item.dataKey} className="comparison-tooltip-row"><span><i style={{background:item.color}}/>{formatLabel?.(item) ?? humanize(item.name)}</span>
          <b>{formatValue ? formatValue(item.value,item.dataKey) : displayValue(item.value)}</b>
          <b className="comparison-tooltip-prior">{prior ? formatValue ? formatValue(prior.value,prior.dataKey) : displayValue(prior.value) : "—"}</b></div>;
      })}</div>
    </div>;
  }
  return (
    <div className="chart-tooltip">
      {label != null && <strong>{categoryLabel(xField, tick(label))}</strong>}
      {ordered.map((item) => (
        <span key={`${item.dataKey}-${item.name}`}>
          <i data-comparison={resolveStyle?.(item)?.type} style={{ color: item.color ?? item.payload?.fill ?? item.fill ?? "var(--chart-1)", background: item.color ?? item.payload?.fill ?? item.fill ?? "var(--chart-1)", opacity:resolveStyle?.(item)?.opacity }} />
          {formatLabel ? formatLabel(item) : humanize(String(item.name ?? item.dataKey))}
          <b>{formatValue ? formatValue(item.value, item.dataKey) : displayValue(item.value)}</b>
        </span>
      ))}
    </div>
  );
}
