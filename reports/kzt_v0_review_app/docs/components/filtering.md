# Filtering and exploration

## Choose the scope

Put `Filters` beside the page title only when the relevant page-wide KPIs and charts respond. Render section-exclusive definitions beside their first affected heading with `SectionHeader` and `useSectionFilters`. For one consumer of a shared query, place `InlineFilters` beside its component and keep selection in local state. View controls belong with the same-scope data controls and do not implicitly reset them; see [controls](controls.md).

Pass scoped reviewed evidence as `sourceRows`; `displayRows` may contain derived plotted aggregates. Never concatenate headline summaries and daily/category aggregates into one source array. Comparisons retain raw evidence for both periods, with the actual derivation in the source definitions. Copying and source inspection must preserve the same scope.

## Page filters

Page-wide filters are controlled by `useDataApp`. `Filters children` places custom controls before generated filters; `trailingControls` places them afterward. Each slot renders once, including when there are no generated filters.

Categorical definitions may set `multiple: true`: selected values are string arrays and `[]` clears the restriction. Selections intersect across page and section scopes. Declared shareable multi-select filters round-trip through validated URL state; `shareInUrl: false` remains private.

Page-wide `<Filters sticky ... />` stays below the protected top bar while scrolling. Place it directly in full-height page content, not a short filter-only wrapper: CSS sticky positioning ends at its containing block. The bar stays transparent before scrolling while chips retain backgrounds; sticky mode compacts its padding without shifting content. Do not implement another scroll-direction handler. Omit `sticky` for section-local controls in `SectionHeader`, and keep page controls outside sortable content. Temporal filters use the shared [date range picker](controls.md#date-and-numeric-inputs), with presets bounded by loaded coverage.

## Tab scope and exploration

`useDashboardTabs([{ id, label, filterIds, defaultFilters, focusFields, aliases, previousLabels }])` opts tabs into independent viewer-local filters. IDs reference snapshot definitions; absent `filterIds` preserves legacy shared filters. Defaults apply per tab. Aliases and previous labels support authored migrations without replacing custom labels. Bindings and private focus are not shared presentation.

`useDataApp()` exposes `exploreDashboard(tabId, { filters, focus })`, `viewFocus`, `setDashboardFocus(focus)`, `canReturnFromExploration`, and `returnFromExploration()`. Only destination-declared fields transfer. Explicit exploration offers Back; normal tab navigation restores browsing state. Keep private entity identity in focus, not URL filters. Reviewed-row functions use the active tab's definitions.

## Section filters

`useSectionFilters(definitions, initialValues?)` keeps local selections independent of page filters and other sections, even when IDs match. Definitions use the same `id`, `label`, `field`, `defaultValue`, `mode`, and optional `queryIds` contract as page filters. It intersects page and section scope against the original reviewed rows before aggregate-row selection. An incompatible page/local selection returns no rows; local All never broadens page scope. Controls do not rerun source queries.

```jsx
function RegionalSection() {
  const { chartProps, chartOverrides } = useDataApp();
  const scope = useSectionFilters([
    { id: "region", label: "Region", field: "region", defaultValue: "all",
      queryIds: ["usage_summary"] },
  ]);
  const chart = chartOverrides["regional-adoption"] ?? { type: "line", x: "week", y: "activeUsers" };
  const scoped = scope.componentProps("usage_summary", [chart.x, chart.series].filter(Boolean));
  return (
    <Section id="regional-section-title" title="Regional adoption"
      filters={<Filters {...scope.filterProps} />}>
      <DataComponent id="regional-adoption" queryId="usage_summary"
        title="Active users over time" kind="chart" chart={chart}
        variant="card" {...scoped}>
        <Chart spec={chart} rows={scoped.displayRows}
          {...chartProps("regional-adoption")} />
      </DataComponent>
    </Section>
  );
}
```

`componentProps(queryId, breakdown?)` supplies `displayRows`, `sourceRows`, and `scopeFilters`, plus `sectionFilters` placement metadata for movable groups. Forward the complete result to **every affected source-backed block** so charts, copied data, source inspection, and the chart editor agree. With `EvidenceChart`, pass `{...scoped}` and `rows={scoped.displayRows}`. Derive captions, totals, and local comparisons from those same rows; do not use unscoped page rows or global prior-period helpers for a locally scoped comparison.

Include temporal/dimension fields in `breakdown` when retaining their full history/categories. Ranged dates otherwise select the latest available scoped endpoint. Section **All dates** selects all dates within page scope. Reset a section selection through its dropdown's **All** / **All dates** option; section filters omit a redundant reset button.

Section selections are local view state, not persisted presentation, URL parameters, or shared-link state. They reset when the section unmounts or reloads. Print retains visible section filter values.

## Filters on movable groups

For movable canvas consumers, provide placement metadata as the hook's third argument:

```jsx
const scope = useSectionFilters(definitions, {}, {
  rowId: "dashboard:engagement",
  componentIds: ["engagement-heatmap", "engagement-scatter"],
  label: "Engagement filters",
});
```

Use `scope.filterProps` in that row's header and `scope.componentProps` on every listed consumer. If a block moves out, its filters appear locally on the card. If an unrelated block joins the filtered row, controls move onto the affected cards instead of implying the newcomer is filtered. Returning to the original group restores the header controls without duplicates. The selection remains shared by that semantic group; dragging never silently changes its data scope. Fixed sections do not need this placement metadata. See [sortable layout](sortable-layout.md) for the placement primitives.

For a new report without visible page-wide controls, author `snapshot.filters: []`; do not copy hidden dashboard filter definitions that silently constrain report data. Report-specific visible-filter bindings are in [reports](reports.md#narrative-and-evidence).
