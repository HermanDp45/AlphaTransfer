import React, { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { ChartRenderer } from "../charting/ChartRenderer.jsx";
import { annotationChartTypes } from "../charting/chart-annotations.js";
import { resolvedChartType } from "../charting/chart-data-shape.js";
import { chartTypes, funnelStageColor, label as humanize, semanticColor } from "../charting/chart-theme.js";
import { Dropdown } from "./Controls.jsx";
import { hexToHsv, hsvToHex, resolvedColorHex, resolvedColorName } from "./chart-color-utils.js";
import { Icon } from "./Icon.jsx";
import { SegmentedControl } from "./SegmentedControl.jsx";
import { Switch } from "./Switch.jsx";

const chartsWithoutAxes = new Set(["pie", "funnel", "sparkline", "rankedList", "sankey"]);
const chartsWithoutOptionalLabels = new Set(["waterfall", "funnel", "sparkline", "heatmap", "boxPlot", "rankedList"]);
const sortableCharts = new Set(["pie", "bar", "horizontalBar", "stackedBar", "stackedBar100",
  "horizontalStackedBar", "horizontalStackedBar100", "rankedList"]);
const chartBaseColors = [
  { label: "Blue", token: "var(--chart-1)" },
  { label: "Purple", token: "var(--chart-2)" },
  { label: "Green", token: "var(--chart-3)" },
  { label: "Orange", token: "var(--chart-4)" },
  { label: "Pink", token: "var(--chart-5)" },
  { label: "Yellow", token: "var(--chart-6)" },
  { label: "Red", token: "var(--chart-7)" },
  { label: "Gray", token: "var(--secondary)" },
];
const chartColorOptions = [
  ...chartBaseColors,
  ...chartBaseColors.slice(0, -1).map(({ label, token }) => ({
    label: `Light ${label.toLowerCase()}`,
    baseToken: token,
    token: `color-mix(in srgb, ${token} 42%, var(--surface))`,
  })),
];
export const chartTypeGroups = [
  { label: "Trends", choices: ["line", "area", "stackedArea", "sparkline"] },
  { label: "Comparisons", choices: ["bar", "horizontalBar", "rankedList", "waterfall"] },
  { label: "Composition", choices: ["stackedBar", "stackedBar100", "horizontalStackedBar", "horizontalStackedBar100", "pie"] },
  { label: "Distribution", choices: ["histogram", "scatter", "heatmap", "boxPlot"] },
  { label: "Flow", choices: ["funnel", "sankey"] },
];
const semanticColorCharts = new Set(["waterfall", "rankedList", "sankey", "heatmap"]);

function optionLabel(value) {
  if (value === "rankedList" || value === "leaderboard") return "Leaderboard";
  return humanize(String(value)).split(" ").map((word, index) =>
    index && !/^[A-Z\d]+$/u.test(word) ? word.toLowerCase() : word).join(" ")
    .replace(/([a-z])100$/u, "$1 (100%)");
}

function PreviewSkeleton() {
  return <div className="explorer-preview-skeleton" aria-hidden="true"><i /><i /><i /><i /></div>;
}

function MountedChart({ onReady, ...props }) {
  useEffect(() => {
    let second;
    const first = requestAnimationFrame(() => { second = requestAnimationFrame(onReady); });
    return () => { cancelAnimationFrame(first); cancelAnimationFrame(second); };
  }, []);
  return <ChartRenderer {...props} />;
}

function ChartColorControl({ colorKey, color, swatchColor = color, onChange }) {
  const name = optionLabel(colorKey);
  const triggerRef = useRef(null);
  const popoverRef = useRef(null);
  const [popoverPosition, setPopoverPosition] = useState(null);
  const [customOpen, setCustomOpen] = useState(false);
  const [themeColorLabels, setThemeColorLabels] = useState({});
  const emittedColor = useRef(null);
  const [customHsv, setCustomHsv] = useState(() => hexToHsv(/^#[\da-f]{6}$/iu.test(color ?? "") ? color : "#0285ff"));
  const customHue = customHsv.hue;
  const [customHex, setCustomHex] = useState(/^#[\da-f]{6}$/iu.test(color ?? "") ? color : "#0285ff");
  const themedColorOptions = chartColorOptions.map(({ label, token, baseToken }) => {
    const actual = themeColorLabels[baseToken ?? token] ?? (baseToken ? label.replace(/^Light /u, "") : label);
    return { token, label: baseToken ? `Light ${actual.toLowerCase()}` : actual };
  });
  const exactColor = themedColorOptions.find((option) => option.token === color)?.label;
  const chartToken = /var\(--chart-(\d+)\)/u.exec(color ?? "");
  const tokenLabel = chartToken ? themeColorLabels[`var(--chart-${chartToken[1]})`]
    ?? chartBaseColors[Number(chartToken[1]) - 1]?.label
    ?? (chartToken[1] === "8" ? "Gray" : undefined) : undefined;
  const colorLabel = exactColor ?? (tokenLabel
    ? color.startsWith("color-mix(") ? `Light ${tokenLabel.toLowerCase()}` : tokenLabel
    : /^#[\da-f]{6}$/iu.test(color ?? "") ? color.toUpperCase()
      : color === "var(--positive)" ? themeColorLabels[color] ?? "Green"
        : color === "var(--negative)" ? themeColorLabels[color] ?? "Red" : "Theme color");

  const customSelected = /^#[\da-f]{6}$/iu.test(color ?? "");
  useEffect(() => {
    const target = triggerRef.current;
    if (!target) return undefined;
    function updateThemeColorLabels() {
      const labels = Object.fromEntries([...chartBaseColors.map(({ token }) => token),
        "var(--chart-8)", "var(--positive)", "var(--negative)"]
        .map((token) => [token, resolvedColorName(token, target)]));
      setThemeColorLabels((current) => Object.keys(labels).every((token) => labels[token] === current[token])
        ? current : labels);
    }
    updateThemeColorLabels();
    const observer = new MutationObserver(updateThemeColorLabels);
    observer.observe(document.documentElement, { attributes: true,
      attributeFilter: ["style", "data-app-theme", "data-color-scheme"] });
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (customSelected && color !== emittedColor.current) {
      setCustomHex(color);
      setCustomHsv(hexToHsv(color));
    }
  }, [color, customSelected]);

  function changeHsv(next) {
    const hex = hsvToHex(next);
    emittedColor.current = hex;
    setCustomHsv(next);
    setCustomHex(hex);
    onChange(hex);
  }

  function toggleCustom() {
    if (!customOpen) {
      const hex = resolvedColorHex(color, triggerRef.current);
      setCustomHex(hex);
      setCustomHsv(hexToHsv(hex));
    }
    setCustomOpen((open) => !open);
  }

  useEffect(() => {
    if (!popoverPosition) return undefined;
    const closeOutside = (event) => {
      if (!popoverRef.current?.contains(event.target) && !triggerRef.current?.contains(event.target)) {
        setPopoverPosition(null);
        setCustomOpen(false);
      }
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        setPopoverPosition(null);
        setCustomOpen(false);
      }
    };
    document.addEventListener("pointerdown", closeOutside);
    window.addEventListener("keydown", closeOnEscape, true);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      window.removeEventListener("keydown", closeOnEscape, true);
    };
  }, [popoverPosition]);

  function togglePopover() {
    if (popoverPosition) {
      setPopoverPosition(null);
      setCustomOpen(false);
      return;
    }
    const bounds = triggerRef.current.getBoundingClientRect();
    setPopoverPosition({
      left: Math.max(12, Math.min(bounds.right - 272, window.innerWidth - 284)),
      top: bounds.bottom + 7,
    });
  }

  function selectArea(event) {
    if (event.type === "pointermove" && event.buttons !== 1) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const saturation = Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100));
    const value = Math.max(0, Math.min(100, (1 - (event.clientY - bounds.top) / bounds.height) * 100));
    changeHsv({ hue: customHue, saturation, value });
    if (event.type === "pointerdown") event.currentTarget.setPointerCapture(event.pointerId);
  }

  return <div className="explorer-color-row"><span className="explorer-property-name">{name}</span>
    <button ref={triggerRef} type="button" className="explorer-color-trigger"
      aria-label={`Color for ${name}`} aria-expanded={Boolean(popoverPosition)} onClick={togglePopover}>
      <i style={{ "--explorer-swatch-color": swatchColor }} /><span>{colorLabel}</span>
    </button>
    {popoverPosition && createPortal(<div ref={popoverRef} className="popover explorer-color-popover"
      aria-label={`Choose color for ${name}`} style={{ left: popoverPosition.left,
        top: Math.min(popoverPosition.top, window.innerHeight - (customOpen ? 325 : 130) - 12) }}>
      <span className="explorer-color-popover-title">Theme colors</span>
      <div className="explorer-color-options" role="group" aria-label="Chart series colors">
        {themedColorOptions.map(({ label, token }) => <button key={token} type="button"
          className="explorer-color-option" aria-label={label} aria-pressed={!customSelected && color === token}
          style={{ "--explorer-swatch-color": token }} onClick={() => onChange(token)} />)}
        <button type="button" className="explorer-color-option explorer-custom-color"
          aria-label={`Custom color for ${name}`} aria-expanded={customOpen} aria-pressed={customSelected}
          style={{ "--explorer-swatch-color": customSelected ? color : "var(--secondary)" }}
          onClick={toggleCustom}><Icon name="plus" size={15} /></button>
      </div>
      {customOpen && <div className="explorer-custom-picker">
        <div className="explorer-color-area" role="slider" aria-label="Color saturation and brightness"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(customHsv.saturation)}
          aria-valuetext={`${Math.round(customHsv.saturation)}% saturation, ${Math.round(customHsv.value)}% brightness`} tabIndex={0}
          style={{ "--explorer-custom-hue": `${customHue}deg` }}
          onPointerDown={selectArea} onPointerMove={selectArea}
          onKeyDown={(event) => {
            if (!event.key.startsWith("Arrow")) return;
            event.preventDefault();
            const step = event.shiftKey ? 10 : 1;
            changeHsv({ ...customHsv,
              saturation: Math.max(0, Math.min(100, customHsv.saturation + (event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0))),
              value: Math.max(0, Math.min(100, customHsv.value + (event.key === "ArrowUp" ? step : event.key === "ArrowDown" ? -step : 0))),
            });
          }}><span className="explorer-color-handle" aria-hidden="true" style={{
            left: `${customHsv.saturation}%`, top: `${100 - customHsv.value}%`, background: customHex,
          }} /></div>
        <input className="explorer-hue-slider" type="range" min="0" max="359"
          aria-label="Color hue" value={customHue} onChange={(event) => {
            const hue = Number(event.target.value);
            changeHsv({ ...customHsv, hue });
          }} />
        <label className="explorer-hex-color"><span>Hex</span>
          <input aria-label={`Custom hex color for ${name}`} value={customHex.toUpperCase()}
            maxLength={7} spellCheck={false} onChange={(event) => {
              const value = event.target.value.startsWith("#") ? event.target.value : `#${event.target.value}`;
              setCustomHex(value);
              if (/^#[\da-f]{6}$/iu.test(value)) onChange(value);
            }} />
        </label>
      </div>}
    </div>, triggerRef.current?.closest(".chart-editor-dialog") ?? document.body)}
  </div>;
}

function BarPresentationControls({ spec, columns, numeric, update }) {
  const options = spec.barOptions ?? {};
  const option = (key, value) => update("barOptions")({ ...options, [key]: value });
  const control = (label, value, onChange, choices = numeric) => <label key={label}><span>{label}</span>
    <Dropdown label={label} value={value} choices={choices} formatChoice={optionLabel} onChange={onChange} /></label>;
  return <div className="explorer-controls" aria-label="Chart controls">
    <section className="explorer-section" aria-label="Reviewed data settings"><h3>Data</h3>
      {control("Category", spec.x, update("x"), columns)}
      {options.series?.length ? options.series.map((series, index) => control(series.label ?? `Measure ${index + 1}`, series.key,
        value => option("series", options.series.map((item, i) => i === index ? { ...item, key: value } : item))))
        : options.range?.length ? options.range.map((field, index) => control(index ? "Range end" : "Range start", field,
          value => option("range", options.range.map((item, i) => i === index ? value : item))))
          : control("Measure", spec.y, update("y"))}
      {options.target && control("Target", options.target, value => option("target", value))}
      {spec.presentation === "rangePosition" && control("Current value", spec.y, update("y"))}
      {options.projection && control("Projection", options.projection, value => option("projection", value))}
      {spec.presentation === "progress" && control("Goal", options.track?.max, value => option("track", { ...options.track, max: value }))}
    </section>
  </div>;
}

export function availableChartTypes(chart) {
  return chartTypes.filter((type) => type !== "leaderboard"
    && (chart.stackable !== false || !type.toLowerCase().includes("stacked")));
}

export function ChartExplorer({
  component, rows, onChange, resolveColor, chartId, visibleSeries, zoomRange, canEdit = true, draft,
}) {
  const columns = useMemo(() => [...new Set(rows.flatMap(Object.keys))], [rows]);
  const availableTypes = availableChartTypes(component.chart);
  const numeric = useMemo(() => columns.filter((column) => rows.some((row) =>
    typeof row[column] === "number" && Number.isFinite(row[column]))), [columns, rows]);
  const categorical = columns.filter((column) => !numeric.includes(column));
  const initial = useMemo(() => ({
    ...component.chart,
    fields: component.chart.fields ?? [component.chart.y],
    series: component.chart.series ?? "",
  }), [component.chart, rows]);
  const [localSpec, setLocalSpec] = useState(initial);
  const [previewReady, setPreviewReady] = useState(false);
  const [previewPainted, setPreviewPainted] = useState(false);
  const spec = useMemo(() => {
    const current = draft ?? localSpec;
    const type = resolvedChartType(current, rows);
    return type === current.type ? current : { ...current, type };
  }, [draft, localSpec, rows]);
  const previewSpec = useDeferredValue(spec);
  const horizontal = spec.type.startsWith("horizontal");
  const categoryFieldLabel = spec.type === "pie" || spec.type === "funnel" ? "Group by"
    : horizontal ? "Y axis" : "X axis";
  const valueFieldLabel = spec.type === "pie" || spec.type === "funnel" ? "Measure"
    : horizontal ? "X axis" : "Y axis";
  const hasLegend = spec.type === "pie" || Boolean(spec.series) || (spec.fields ?? [spec.y]).length > 1;
  const hasAnnotations = annotationChartTypes.includes(spec.type)
    && Array.isArray(spec.annotations) && spec.annotations.length > 0;
  const colorKeys = useMemo(() => semanticColorCharts.has(spec.type) ? []
    : spec.type === "pie" || spec.type === "funnel"
      ? [...new Set(rows.map((row) => row[spec.x]).filter((value) => value != null))]
      : spec.series ? [...new Set(rows.map((row) => row[spec.series]).filter((value) => value != null))]
        : (spec.fields ?? [spec.y]).filter(Boolean), [rows, spec.fields, spec.series, spec.type, spec.x, spec.y]);
  const colorForKey = (key, index) => spec.colors?.[key]
    ?? (spec.type === "funnel" ? spec.colors?.[spec.y]
      ?? resolveColor?.({ field: spec.y, index: 0 }) ?? semanticColor({ field: spec.y, index: 0 })
      : ["line", "sparkline"].includes(spec.type) && /(?:target|plan|forecast|projected|benchmark)/iu.test(String(key))
      ? "var(--secondary)" : resolveColor?.(spec.series || spec.type === "pie" || spec.type === "funnel"
      ? { field: spec.y, dimension: spec.series || spec.x, value: key, index }
      : { field: key, index }));

  useEffect(() => {
    let revealFrame;
    const shellFrame = requestAnimationFrame(() => {
      revealFrame = requestAnimationFrame(() => setPreviewReady(true));
    });
    return () => {
      cancelAnimationFrame(shellFrame);
      if (revealFrame) cancelAnimationFrame(revealFrame);
    };
  }, []);

  const update = (field) => (value) => {
    const current = draft ?? localSpec;
    const next = {
      ...current,
      [field]: value,
      ...(field === "y" ? { fields: [value] } : {}),
      ...(field === "x" && value === spec.series ? { series: "" } : {}),
      ...(field === "xLabel" ? { showXAxisLabel: Boolean(value.trim()) } : {}),
      ...(field === "yLabel" ? { showYAxisLabel: Boolean(value.trim()) } : {}),
    };
    setLocalSpec(next);
    onChange?.(next, JSON.stringify(next) !== JSON.stringify(initial), field);
  };

  return <div className="chart-explorer">
    <div className="explorer-preview">
      <div className="explorer-chart" data-ready={previewPainted} aria-busy={!previewPainted}>
        <PreviewSkeleton />
        <div className="explorer-chart-content">{previewReady && <MountedChart onReady={() => setPreviewPainted(true)} spec={previewSpec} rows={rows} height={500}
        chartId={chartId ?? component.id} resolveColor={resolveColor}
        visibleSeries={visibleSeries} zoomRange={zoomRange} />}</div>
      </div>
    </div>
    {canEdit && (spec.presentation ? <BarPresentationControls spec={spec} columns={columns} numeric={numeric} update={update} />
      : <div className="explorer-controls" aria-label="Chart controls">
      <section className="explorer-section explorer-visualization-section" aria-label="Visualization settings">
        <h3>Visualization</h3>
        <label><span>Chart type</span><Dropdown label="Chart type" value={spec.type}
          choices={availableTypes} groups={chartTypeGroups.map((group) => ({ ...group,
            choices: group.choices.filter((type) => availableTypes.includes(type)) })).filter((group) => group.choices.length)}
          contentClassName="chart-type-menu" formatChoice={optionLabel} onChange={update("type")} /></label>
      </section>
      <section className="explorer-section" aria-label="Reviewed data settings">
        <h3>Data</h3>
        <label><span>{categoryFieldLabel}</span><Dropdown label={categoryFieldLabel} value={spec.x}
          choices={columns} formatChoice={optionLabel} onChange={update("x")} /></label>
        <label><span>{valueFieldLabel}</span><Dropdown label={valueFieldLabel} value={spec.y}
          choices={numeric} formatChoice={optionLabel} onChange={update("y")} /></label>
        <label><span>Split by</span><Dropdown label="Split series by" value={spec.series}
          choices={["", ...categorical.filter((field) => field !== spec.x)]}
          formatChoice={optionLabel} onChange={update("series")} /></label>
        {sortableCharts.has(spec.type) && <label><span>Sort order</span>
          <Dropdown label="Sort order" value={spec.sortOrder ?? (spec.type === "rankedList" ? "descending" : "original")}
            choices={["original", "ascending", "descending"]} formatChoice={optionLabel}
            onChange={update("sortOrder")} /></label>}
      </section>
      {!chartsWithoutAxes.has(spec.type) && <section className="explorer-section" aria-label="Chart axes settings">
        <h3>Axes</h3>
        {!spec.series && !spec.barFields?.length && (spec.fields ?? []).length > 1
          && ["line", "bar", "area", "horizontalBar"].includes(spec.type) &&
          <label><span>{horizontal ? "Secondary axis" : "Right axis"}</span>
            <Dropdown label="Secondary axis" value={spec.rightAxisFields === undefined ? "auto"
              : spec.rightAxisFields[0] ?? "none"} choices={["auto", "none", ...spec.fields]}
              formatChoice={(value) => value === "auto" ? "Automatic" : value === "none" ? "Shared scale" : optionLabel(value)}
              onChange={(value) => update("rightAxisFields")(value === "auto" ? undefined : value === "none" ? [] : [value])} />
          </label>}
        <label><span>X axis label</span><input aria-label="X axis title" value={spec.xLabel ?? ""}
          onChange={(event) => update("xLabel")(event.target.value)} /></label>
        {!spec.type.startsWith("horizontal") && <label><span>Y axis label</span>
          <input aria-label="Y axis title" value={spec.yLabel ?? ""}
            onChange={(event) => update("yLabel")(event.target.value)} /></label>}
        <div className="explorer-control-row"><span>Y axis side</span>
          <SegmentedControl ariaLabel="Y axis side" value={spec.yAxisPosition ?? "left"}
            options={[{ value: "left", label: "Left" }, { value: "right", label: "Right" }]}
            onChange={update("yAxisPosition")} fullWidth />
        </div>
        {spec.type !== "heatmap" && <Switch label="Start axis at zero" fullWidth checked={spec.startAtZero !== false}
          onChange={update("startAtZero")} />}
      </section>}
      {(colorKeys.length > 0 || spec.type === "heatmap") && <section className="explorer-section" aria-label="Chart color settings">
        <h3>Colors</h3>
        {spec.type === "heatmap" && <ChartColorControl colorKey="Base color"
          color={spec.baseColor ?? "var(--chart-1)"} onChange={update("baseColor")} />}
        {colorKeys.map((key, index) => <ChartColorControl key={String(key)} colorKey={key}
          color={colorForKey(key, index)}
          swatchColor={spec.type === "funnel" ? funnelStageColor(colorForKey(key, index),
            rows.findIndex((row) => row[spec.x] === key), rows.length) : undefined}
          onChange={(color) => update("colors")({ ...spec.colors, [key]: color })} />)}
      </section>}
      {(!chartsWithoutOptionalLabels.has(spec.type) || hasLegend || hasAnnotations) &&
        <section className="explorer-section" aria-label="Chart appearance settings">
          <h3>Appearance</h3>
          {!chartsWithoutOptionalLabels.has(spec.type) && <Switch label="Show values" fullWidth
            checked={spec.showValues === true} onChange={update("showValues")} />}
          {hasLegend && <Switch label="Show legend" fullWidth
            checked={spec.showLegend !== false} onChange={update("showLegend")} />}
          {hasAnnotations && <Switch label="Show annotations" fullWidth
            checked={spec.showAnnotations !== false} onChange={update("showAnnotations")} />}
        </section>}
    </div>)}
  </div>;
}
