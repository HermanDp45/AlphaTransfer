import { useState } from "react";

import { useDataAppShell } from "./DataAppContext.jsx";
import { resolveSectionRows } from "./use-data-app.js";

/** Local view state only: never writes page filters, URL state, or reviewed rows. */
export function useSectionFilters(definitions = [], initialValues = {}, sectionScope) {
  const { snapshot, queries, filters: pageValues } = useDataAppShell();
  const [selected, setSelected] = useState(initialValues);
  const values = Object.fromEntries(definitions.map(({ id, defaultValue }) => [id, selected[id] ?? defaultValue ?? "all"]));
  const setFilter = (id, value) => {
    if (definitions.some((definition) => definition.id === id)) setSelected((previous) => ({ ...previous, [id]: value }));
  };
  // Share one resolution between the renderer and its source/copy metadata.
  // The cache lives for this render only, so changed scopes cannot reuse stale rows.
  const resolved = new Map();
  const reviewedRows = (queryId, breakdown = []) => {
    const key = JSON.stringify([queryId, breakdown]);
    if (!resolved.has(key)) resolved.set(key, resolveSectionRows(
      queries[queryId]?.rows ?? [], snapshot.filters ?? [], pageValues,
      definitions, values, queryId, breakdown,
    ));
    return resolved.get(key);
  };
  const filterProps = { filters: definitions, queries, values, onChange: setFilter,
    ariaLabel: sectionScope?.label ?? "Section filters", showClear: false,
    dateAllValue: "all", sticky: false, sectionScope };
  const componentProps = (queryId, breakdown = []) => {
    const rows = reviewedRows(queryId, breakdown);
    return {
      displayRows: rows, sourceRows: rows,
      ...(sectionScope ? { sectionFilters: filterProps } : {}),
      scopeFilters: definitions.filter(({ id, queryIds }) => values[id] !== "all"
        && (!Array.isArray(values[id]) || values[id].length > 0)
        && (!Array.isArray(queryIds) || queryIds.includes(queryId)))
        .map(({ id, field, label }) => ({ field, label, value: values[id] })),
    };
  };
  return { values, setFilter, reviewedRows, componentProps, filterProps };
}
