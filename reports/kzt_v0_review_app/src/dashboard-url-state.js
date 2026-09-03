import { isTemporalField } from "./source-provenance.js";

const filterParameterPrefix = "f.";
const maximumFilterIdLength = 200;
const maximumFilterValueLength = 2_000;

function shareableFilterDefinitions(snapshot) {
  return (snapshot?.filters ?? [])
    .filter(({ id, shareInUrl }) => typeof id === "string" && id.length > 0
      && id.length <= maximumFilterIdLength && shareInUrl !== false)
    .sort((left, right) => left.id.localeCompare(right.id));
}

function filterChoices(snapshot, filter) {
  const queries = Array.isArray(filter.queryIds)
    ? filter.queryIds.map((id) => snapshot?.queries?.[id]).filter(Boolean)
    : Object.values(snapshot?.queries ?? {});
  return new Set(queries.flatMap(({ rows = [] }) => rows
    .map((row) => row?.[filter.field])
    .filter((value) => value !== undefined && value !== null && String(value).length > 0)
    .map(String)));
}

export function validFilterValue(snapshot, filter, value) {
  if (Array.isArray(value)) {
    if (!filter.multiple || isTemporalField(filter.field, filter.type ?? filter.valueType)
      || value.length > 100 || JSON.stringify(value).length > maximumFilterValueLength) return false;
    const choices = filterChoices(snapshot, filter);
    return value.every(item => typeof item === "string" && item !== "all" && choices.has(item));
  }
  if (typeof value !== "string" || value.length > maximumFilterValueLength) return false;
  if (value !== "all") {
    const choices = filterChoices(snapshot, filter);
    if (filter.mode === "through" && /^\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}$/.test(value)) {
      const [start, end] = value.split("..");
      const validDay = day => /^\d{4}-\d{2}-\d{2}$/.test(day) && Number.isFinite(Date.parse(day))
        && new Date(day).toISOString().slice(0, 10) === day;
      const dates = [...choices].filter(validDay).sort();
      // Calendar presets can start or end between sparse observations. Preserve
      // their exact scope while keeping both boundaries inside reviewed coverage.
      return validDay(start) && validDay(end) && start <= end && dates.length > 0
        && start >= dates[0] && end <= dates.at(-1);
    }
    return choices.has(value);
  }
  return !isTemporalField(filter.field, filter.type ?? filter.valueType)
    || !filter.defaultValue || filter.defaultValue === "all";
}

function locationUrl(location) {
  try {
    return new URL(typeof location === "string" ? location : location?.href ?? "");
  } catch {
    return null;
  }
}

export function readDashboardUrlState(snapshot, tabs = [], location = globalThis.location) {
  const url = locationUrl(location);
  if (!url) return { tab: null, filters: {}, hasViewState: false };

  const requestedTabs = url.searchParams.getAll("tab");
  const requestedTab = requestedTabs.length === 1 ? requestedTabs[0] : null;
  const tab = requestedTab !== null && tabs.some(({ id }) => id === requestedTab) ? requestedTab : null;
  const filterEntries = [];

  for (const filter of shareableFilterDefinitions(snapshot)) {
    const parameter = `${filterParameterPrefix}${filter.id}`;
    if (!url.searchParams.has(parameter)) continue;
    const values = url.searchParams.getAll(parameter);
    if (values.length !== 1 || values[0].length > maximumFilterValueLength) continue;
    let value = values[0];
    if (filter.multiple && value.startsWith("[")) {
      try { value = JSON.parse(value); } catch { continue; }
    }
    if (!validFilterValue(snapshot, filter, value)) continue;
    filterEntries.push([filter.id, value]);
  }

  return {
    tab,
    filters: Object.fromEntries(filterEntries),
    hasViewState: tab !== null || filterEntries.length > 0,
  };
}

export function resolveDashboardUrlFilters(snapshot, localFilters = {}, state, { reset = false } = {}) {
  if (!state?.hasViewState && !reset) return { ...localFilters };

  const resolved = new Map(Object.entries(localFilters));
  for (const filter of shareableFilterDefinitions(snapshot)) {
    resolved.set(filter.id, Object.hasOwn(state?.filters ?? {}, filter.id)
      ? state.filters[filter.id] : filter.defaultValue ?? "all");
  }
  return Object.fromEntries(resolved);
}

export function serializeDashboardUrlState(snapshot, tabs = [], { tab, filters = {} } = {}) {
  const parameters = new URLSearchParams();
  if (tab && tabs.some(({ id }) => id === tab) && tab !== tabs[0]?.id) {
    parameters.set("tab", tab);
  }

  for (const filter of shareableFilterDefinitions(snapshot)) {
    const value = filters[filter.id];
    const encode = selection => Array.isArray(selection)
      ? JSON.stringify([...new Set(selection)].sort()) : String(selection);
    const defaultValue = filter.defaultValue ?? (filter.multiple ? [] : "all");
    if (value === undefined || encode(value) === encode(defaultValue) || !validFilterValue(snapshot, filter, value)) continue;
    parameters.set(`${filterParameterPrefix}${filter.id}`, encode(value));
  }

  return parameters;
}

export function dashboardUrlWithState(location, snapshot, tabs, state, { preserveExisting = true } = {}) {
  const url = locationUrl(location);
  if (!url) return null;

  if (!preserveExisting) {
    url.username = "";
    url.password = "";
    url.search = "";
    url.hash = "";
  } else {
    for (const key of [...url.searchParams.keys()]) {
      if (key === "tab" || key.startsWith(filterParameterPrefix)) url.searchParams.delete(key);
    }
  }

  for (const [key, value] of serializeDashboardUrlState(snapshot, tabs, state)) {
    url.searchParams.set(key, value);
  }
  return url;
}
