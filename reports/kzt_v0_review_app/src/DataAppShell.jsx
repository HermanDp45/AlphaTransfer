import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { semanticColorResolver } from "./charting/chart-theme.js";
import { resolveChartSpec } from "./charting/chart-overrides.js";
import { canDownloadChartImage, copyChartImage } from "./chart-image.js";
import {
  canonicalDashboardPath,
  chartPermalink,
  componentPermalink,
  componentPermalinkId,
  componentPermalinkShortId,
  isComponentPermalinkTargetVisible,
  readComponentPermalink,
  validComponentId,
} from "./chart-permalink.js";
import { ChartExplorer } from "./components/ChartExplorer.jsx";
import { ChartExportDialog } from "./components/ChartExportDialog.jsx";
import { chartExportMetadata } from "./chart-export.js";
import { chartColorContext } from "./components/chart-color-utils.js";
import { DashboardAskProvider } from "./components/DashboardAsk.jsx";
import { DataAppThemeDrawer, DataAppTopbar } from "./components/DataAppChrome.jsx";
import { DataAppToast } from "./components/DataAppToast.jsx";
import { Icon } from "./components/Icon.jsx";
import { SourceSidebar } from "./components/SourceInspector.jsx";
import { Dialog, MenuItem, Tooltip } from "./components/ui.jsx";
import { dataAppActionHref, printDataApp, submitDataAppAction } from "./data-app-actions.js";
import { DataAppBlockLayoutContext, DataAppContext } from "./DataAppContext.jsx";
import { canUseDashboardAsk } from "./dashboard-ask.js";
import { dashboardTabId, dashboardTabSnapshot, dashboardView, drillDashboardView } from "./dashboard-view-state.js";
import { dashboardUrlWithState, readDashboardUrlState, resolveDashboardUrlFilters } from "./dashboard-url-state.js";
import {
  normalizeAppearance,
  normalizePresentation,
  editHistoryValue,
  presentationBeforeEdits,
  readLocalPresentation,
  readViewerAppearance,
  writeViewerAppearance,
} from "./presentation-state.js";
import { currentDataAppReference, dataAppPromptTarget } from "./runtime-environment.js";
import { reviewedComponentClipboard } from "./source-provenance.js";
import { applyDataAppTheme, dataAppThemeTokens } from "./theme-presets.js";
import { setDataAppAppearance } from "./theme-runtime.js";
import { useDataApp } from "./use-data-app.js";
import { useDataAppImageTools } from "./use-data-app-image-tools.js";
import { editableTextTarget, useInlineEditing } from "./use-inline-editing.js";
import { usePresentationHistory, usePresentationPersistence } from "./use-presentation.js";
import { usePendingCodeChanges } from "./use-publication.js";

const defaultDashboardTabs = [{ id: "dashboard", label: "Dashboard" }];

// This child mounts when the menu opens, after the chart has rendered. Use the
// same capture capability as quick copy/download, including resolved chart types.
function ChartExportAction({ component, onSelect }) {
  return canDownloadChartImage(component.id)
    ? <MenuItem icon="download" onSelect={onSelect}>Export chart</MenuItem> : null;
}

function reconcileDashboardTabs(current, authored) {
  const remainingById = new Map(authored.map((tab) => [tab.id, tab]));
  const next = current.flatMap(({ id, label }) => {
    const definition = remainingById.get(id);
    if (!definition) return [];
    remainingById.delete(id);
    return [{ id: definition.id, label: definition.previousLabels?.includes(label) || defaultDashboardTabs.some(tab => tab.id === id && tab.label === label) ? definition.label : label }];
  });
  for (const tab of authored) {
    if (!remainingById.has(tab.id)) continue;
    next.push({ id: tab.id, label: tab.label });
    remainingById.delete(tab.id);
  }
  return next;
}

// Keep opening the editor independent of the dashboard render. Its real header
// mounts synchronously; only data resolution and the expensive body are deferred.
const ChartEditorHost = forwardRef(function ChartEditorHost(
  { reviewedRows, resolveColor, chartStates, canEdit, onClose, onSave },
  ref,
) {
  const [request, setRequest] = useState(null);
  useImperativeHandle(
    ref,
    () => ({
      open(next, immediate = true) {
        if (immediate) flushSync(() => setRequest(next));
        else setRequest(next);
      },
      close() {
        setRequest(null);
      },
      followPermalink(next, matches) {
        setRequest((current) =>
          current?.permalink && (!next?.detail || !matches(current.component.id, next.id)) ? null : current,
        );
      },
    }),
    [],
  );
  if (!request) return null;
  const { component } = request;
  return (
    <ChartEditorDialog
      key={component.id}
      component={component}
      getRows={() =>
        component.displayRows ??
        reviewedRows(component.queryId, [component.chart.x, component.chart.series].filter(Boolean))
      }
      resolveColor={resolveColor}
      visibleSeries={chartStates[component.id]?.visibleSeries}
      zoomRange={chartStates[component.id]?.zoomRange}
      canEdit={canEdit}
      onClose={() => onClose(request)}
      onSave={(spec) => onSave(spec, request)}
    />
  );
});

// Source data is resolved after the drawer has painted, without rerendering the dashboard.
const SourceSidebarHost = forwardRef(function SourceSidebarHost(
  { queries, reviewedRows, activeFilters, chartStates, snapshot },
  ref,
) {
  const [component, setComponent] = useState(null);
  useImperativeHandle(
    ref,
    () => ({
      open(next) {
        flushSync(() => setComponent(next));
      },
      close() {
        setComponent(null);
      },
      update(next) {
        setComponent((current) => current?.id === next.id ? next : current);
      },
    }),
    [],
  );
  const getSource = useCallback((queryId = component?.queryId) => {
    if (!component) return null;
    const selectedInlineFilters = Object.entries(chartStates[component.id]?.inlineFilters ?? {})
      .filter(
        ([field, value]) =>
          value !== "all" && reviewedRows(queryId, [field]).some((row) => row[field] === value),
      )
      .map(([field, value]) => ({
        field,
        value,
        label:
          field === "segment" ? "Product" : snapshot.filters?.find((filter) => filter.field === field)?.label ?? field,
      }));
    return {
      query: queries[queryId],
      rows:
        component.sourceRowsByQuery?.[queryId] ??
        (queryId === component.queryId ? component.sourceRows : undefined) ??
        reviewedRows(queryId, [component.chart?.x, component.chart?.series].filter(Boolean)),
      filters: [...activeFilters.filter((filter) => !Array.isArray(filter.queryIds) || filter.queryIds.includes(queryId)), ...(component.scopeFilters ?? []), ...selectedInlineFilters],
    };
  }, [component, queries, reviewedRows, activeFilters, chartStates, snapshot]);
  return (
    component && (
      <SourceSidebar
        key={component.id}
        component={component}
        queries={queries}
        getSource={getSource}
        onClose={() => setComponent(null)}
      />
    )
  );
});

function ChartEditorDialog({ component, getRows, resolveColor, visibleSeries, zoomRange, canEdit, onClose, onSave }) {
  const [colorContext] = useState(() => {
    const source = document.querySelector(`[data-component-id="${CSS.escape(component.id)}"]`);
    return chartColorContext(component.chart, source ? getComputedStyle(source) : null);
  });
  const [editor, setEditor] = useState(() => ({
    draft: component.chart,
    history: [component.chart],
    historyIndex: 0,
    dirty: false,
    lastField: null,
  }));
  const [closing, setClosing] = useState(false);
  const [contentReady, setContentReady] = useState(false);
  const closeTimer = useRef(null);

  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  useEffect(() => {
    let secondFrame;
    const frame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => setContentReady(true));
    });
    return () => {
      window.cancelAnimationFrame(frame);
      window.cancelAnimationFrame(secondFrame);
    };
  }, []);
  const rows = useMemo(() => (contentReady ? getRows() : []), [contentReady, getRows]);

  function dismiss(action = onClose) {
    if (closing) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      action();
      return;
    }
    setClosing(true);
    closeTimer.current = window.setTimeout(action, 110);
  }

  function stepHistory(direction) {
    setEditor((current) => {
      const historyIndex = current.historyIndex + direction;
      if (historyIndex < 0 || historyIndex >= current.history.length) return current;
      const draft = current.history[historyIndex];
      return {
        ...current,
        draft,
        historyIndex,
        lastField: null,
        dirty: JSON.stringify(draft) !== JSON.stringify(current.history[0]),
      };
    });
  }

  useEffect(() => {
    if (!canEdit) return undefined;
    const handleKeyDown = (event) => {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      if (event.key.toLowerCase() === "z") {
        event.preventDefault();
        stepHistory(event.shiftKey ? 1 : -1);
      } else if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        if (editor.dirty) dismiss(() => onSave(editor.draft));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [canEdit, editor.dirty, editor.draft, onSave]);

  return (
    <Dialog
      title={component.title}
      expanded
      initialFocusSelector=".dialog-header"
      style={colorContext}
      backdropClassName="chart-editor-backdrop"
      className={`chart-editor-dialog${closing ? " is-closing" : ""}`}
      showClose={!canEdit}
      headerActions={
        canEdit && (
          <>
            <div className="chart-editor-history" role="group" aria-label="Chart edit history">
              <span className="history-tooltip-trigger">
                <button
                  type="button"
                  className="chart-editor-icon"
                  aria-label="Undo chart change"
                  disabled={editor.historyIndex === 0}
                  onClick={() => stepHistory(-1)}
                >
                  <Icon name="undo" size={18} />
                </button>
                <Tooltip className="topbar-mode-tooltip">Undo</Tooltip>
              </span>
              <span className="history-tooltip-trigger">
                <button
                  type="button"
                  className="chart-editor-icon"
                  aria-label="Redo chart change"
                  disabled={editor.historyIndex >= editor.history.length - 1}
                  onClick={() => stepHistory(1)}
                >
                  <Icon name="undo" size={18} className="chart-editor-redo-icon" />
                </button>
                <Tooltip className="topbar-mode-tooltip">Redo</Tooltip>
              </span>
            </div>
            <button type="button" className="button ghost chart-editor-cancel" onClick={() => dismiss()}>
              Cancel
            </button>
            <button
              type="button"
              className="button primary chart-editor-save"
              disabled={!editor.dirty}
              onClick={() => dismiss(() => onSave(editor.draft))}
            >
              Save
            </button>
          </>
        )
      }
      onClose={() => dismiss()}
    >
      {contentReady ? (
        <ChartExplorer
          component={component}
          rows={rows}
          chartId={component.id}
          resolveColor={resolveColor}
          visibleSeries={visibleSeries}
          zoomRange={zoomRange}
          draft={editor.draft}
          canEdit={canEdit}
          onChange={
            canEdit
              ? (spec, dirty, field) =>
                  setEditor((current) => {
                    const continuous = ["xLabel", "yLabel"].includes(field) && current.lastField === field;
                    const history = continuous
                      ? [...current.history.slice(0, current.historyIndex), spec]
                      : [...current.history.slice(0, current.historyIndex + 1), spec];
                    return { draft: spec, dirty, history, historyIndex: history.length - 1, lastField: field };
                  })
              : undefined
          }
        />
      ) : (
        <div className="chart-explorer editor-loading-layout" aria-label="Loading chart editor">
          <div className="explorer-preview">
            <div className="explorer-preview-skeleton">
              <i />
              <i />
              <i />
              <i />
            </div>
          </div>
          <div className="explorer-controls editor-controls-skeleton">
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
      )}
    </Dialog>
  );
}

export function DataAppShell({
  snapshot,
  hosted,
  canEdit = true,
  onSnapshotChange,
  initialPresentation = {},
  initialRevision = 0,
  children,
}) {
  const reportSurface = snapshot.surface === "report";
  useEffect(() => {
    // Focus restoration after a pointer-operated menu must not paint a keyboard ring.
    // Keep this on the document so portalled menus/dialogs share the same modality.
    const root = document.documentElement;
    const pointer = () => { root.dataset.inputModality = "pointer"; };
    const keyboard = event => {
      if (!event.metaKey && !event.ctrlKey && !event.altKey
        && !["Shift", "Control", "Alt", "Meta"].includes(event.key)) root.dataset.inputModality = "keyboard";
    };
    document.addEventListener("pointerdown", pointer, true);
    document.addEventListener("keydown", keyboard, true);
    return () => {
      document.removeEventListener("pointerdown", pointer, true);
      document.removeEventListener("keydown", keyboard, true);
      delete root.dataset.inputModality;
    };
  }, []);
  const surfaceNoun = reportSurface ? "report" : "dashboard";
  const [savedPresentation] = useState(() => {
    const personal = readLocalPresentation(snapshot);
    const normalized = normalizePresentation(
      hosted
        ? {
            ...initialPresentation,
            ...(personal.filters ? { filters: personal.filters } : {}),
            ...(personal.assumptions ? { assumptions: personal.assumptions } : {}),
            ...(personal.tabViews ? { tabViews: personal.tabViews } : {}),
          }
        : personal,
    );
    if (reportSurface) {
      delete normalized.tabs;
      delete normalized.refreshSchedule;
    }
    return normalized;
  });
  const [initialDashboardHref] = useState(() => globalThis.location?.href);
  const [initialUrlState] = useState(() =>
    reportSurface
      ? { tab: null, filters: {}, hasViewState: false }
      : readDashboardUrlState(snapshot, defaultDashboardTabs),
  );
  const sourceSidebarRef = useRef(null);
  const chartEditorRef = useRef(null);
  const [mode, setMode] = useState("view");
  const [chartExport, setChartExport] = useState(null);
  const [editSession, setEditSession] = useState(null);
  const editSessionRef = useRef(editSession);
  editSessionRef.current = editSession;
  const [chartOverrides, setChartOverrides] = useState(savedPresentation.chartOverrides ?? {});
  const [blockLayouts, setBlockLayouts] = useState(savedPresentation.blockLayouts ?? {});
  const [hidden, setHidden] = useState(() => new Set(savedPresentation.hiddenBlocks ?? []));
  const [actionStatus, setActionStatus] = useState("");
  const [activeTheme, setActiveTheme] = useState(savedPresentation.theme ?? "original");
  const [appearance, setAppearance] = useState(normalizeAppearance(savedPresentation.appearance));
  const [viewerAppearance, setViewerAppearance] = useState(() => readViewerAppearance(snapshot));
  const [chartStates, setChartStates] = useState({});
  const [themesOpen, setThemesOpen] = useState(false);
  const [originalThemeTokens] = useState(() => {
    if (typeof document === "undefined" || typeof getComputedStyle !== "function") {
      return dataAppThemeTokens.map(() => "");
    }
    const computed = getComputedStyle(document.documentElement);
    return dataAppThemeTokens.map((token) => computed.getPropertyValue(`--${token}`).trim());
  });
  const [appTitle, setAppTitle] = useState(savedPresentation.title ?? snapshot.title);
  const [verification, setVerification] = useState(savedPresentation.verification);
  const [pendingVerification, setPendingVerification] = useState(null);
  const [narrativeEdits, setNarrativeEdits] = useState(savedPresentation.textEdits ?? {});
  const [componentTitles, setComponentTitles] = useState(savedPresentation.componentTitles ?? {});
  const [tabs, setTabs] = useState(savedPresentation.tabs ?? defaultDashboardTabs);
  const [activeTabId, setActiveTabId] = useState(
    () => initialUrlState.tab ?? (savedPresentation.tabs ?? defaultDashboardTabs)[0].id,
  );
  const [tabDefinitions, setTabDefinitions] = useState([]);
  const tabDefinitionsRef = useRef(tabDefinitions);
  tabDefinitionsRef.current = tabDefinitions;
  const scopedTabs = tabDefinitions.some(tab => Array.isArray(tab.filterIds));
  const scopedSnapshot = useMemo(() => dashboardTabSnapshot(snapshot, tabDefinitions, activeTabId),
    [snapshot, tabDefinitions, activeTabId]);
  const [tabViews, setTabViews] = useState(savedPresentation.tabViews ?? {});
  const tabViewsRef = useRef(tabViews);
  tabViewsRef.current = tabViews;
  const [viewFocus, setViewFocus] = useState({});
  const [drillReturn, setDrillReturn] = useState(null);
  const viewRef = useRef({});
  const { queries, filters, setFilter, replaceFilters, reviewedRows, reviewedPeriodRows, reviewedAggregatePeriodRows, activeFilters } = useDataApp(snapshot, {
    filterDefinitions: scopedSnapshot.filters,
    visibleFilterIds: snapshot.report?.visibleFilterIds ?? snapshot.visibleReportFilters,
    hosted,
    canEdit,
    onSnapshotChange,
    initialFilters: resolveDashboardUrlFilters(snapshot, savedPresentation.filters, initialUrlState),
    authoritativeInitialFilters: initialUrlState.filters,
  });
  viewRef.current = { tabId: activeTabId, filters, focus: viewFocus, returnTo: drillReturn };
  const activeTabRef = useRef(activeTabId);
  const tabsRef = useRef(tabs);
  const filtersRef = useRef(filters);
  activeTabRef.current = activeTabId;
  tabsRef.current = tabs;
  filtersRef.current = filters;
  const [tabRegistrationReady, setTabRegistrationReady] = useState(false);
  const dashboardBusyRef = useRef(false);
  const [dashboardBusy, updateDashboardBusy] = useState(false);
  const setDashboardBusy = useCallback(value => {
    dashboardBusyRef.current = Boolean(value);
    updateDashboardBusy(Boolean(value));
  }, []);
  const initialAuthoredTabsRegistered = useRef(false);
  const registerDashboardTabs = useCallback(
    (definitions) => {
      if (reportSurface || !Array.isArray(definitions)) return;
      const uniqueIds = new Set();
      const normalized = definitions.slice(0, 20).flatMap((entry) => {
        const id = typeof entry?.id === "string" ? entry.id.trim() : "";
        const label = typeof entry?.label === "string" ? entry.label.trim() : "";
        if (!id || id.length > 80 || !/^[a-zA-Z0-9_-]+$/u.test(id) || !label || label.length > 100 || uniqueIds.has(id))
          return [];
        uniqueIds.add(id);
        return [{ id, label, ...(Array.isArray(entry.previousLabels) ? { previousLabels: entry.previousLabels } : {}) }];
      });
      const authored = normalized.length ? normalized : defaultDashboardTabs;
      const bindings = authored.map(tab => ({ ...definitions.find(entry => entry.id === tab.id), ...tab }));
      tabDefinitionsRef.current = bindings;
      setTabDefinitions(current => JSON.stringify(current) === JSON.stringify(bindings) ? current : bindings);
      if (!normalized.length) {
        uniqueIds.clear();
        uniqueIds.add("dashboard");
      }
      const nextTabs = reconcileDashboardTabs(tabsRef.current, authored);
      if (!initialAuthoredTabsRegistered.current) {
        initialAuthoredTabsRegistered.current = true;
        const requested = initialDashboardHref ? new URL(initialDashboardHref).searchParams.get("tab") : null;
        const tabId = dashboardTabId(bindings, requested ?? activeTabRef.current);
        const scoped = dashboardTabSnapshot(snapshot, bindings, tabId);
        const state = readDashboardUrlState(scoped, authored, initialDashboardHref);
        const historyView = globalThis.history?.state?.dataAppView;
        const restored = historyView && historyView.artifactId === snapshot.id && historyView.tabId === tabId ? historyView : null;
        const saved = restored ?? savedPresentation.tabViews?.[tabId] ?? { filters: savedPresentation.filters };
        const view = dashboardView(snapshot, bindings, tabId, saved, state);
        setActiveTabId(tabId);
        replaceFilters(view.filters);
        setViewFocus(view.focus);
        setDrillReturn(restored?.returnTo ?? null);
      } else if (!uniqueIds.has(activeTabRef.current)) {
        setActiveTabId(nextTabs[0].id);
      }
      setTabs((current) => {
        const next = reconcileDashboardTabs(current, authored);
        return next.length === current.length &&
          next.every((tab, index) => tab.id === current[index].id && tab.label === current[index].label)
          ? current
          : next;
      });
      setTabRegistrationReady(true);
    },
    [initialDashboardHref, replaceFilters, reportSurface, savedPresentation, snapshot],
  );
  useEffect(() => {
    if (reportSurface || initialAuthoredTabsRegistered.current) {
      setTabRegistrationReady(true);
      return;
    }
    const state = readDashboardUrlState(snapshot, defaultDashboardTabs, initialDashboardHref);
    setTabs(defaultDashboardTabs);
    setActiveTabId(state.tab ?? "dashboard");
    replaceFilters(resolveDashboardUrlFilters(snapshot, savedPresentation.filters, state));
    setTabRegistrationReady(true);
  }, [initialDashboardHref, replaceFilters, reportSurface, savedPresentation, snapshot]);
  const [linkedComponent, setLinkedComponent] = useState(() =>
    hosted ? readComponentPermalink(globalThis.location) : null,
  );
  const componentTargets = useRef(new Map());
  const handledComponentPermalink = useRef(null);
  const unavailableComponentPermalink = useRef(null);
  const componentTabProbe = useRef(null);
  const componentHighlight = useRef(null);
  const registerComponent = useCallback((component, element) => {
    if (element) {
      componentTargets.current.set(component.id, { component, element });
      sourceSidebarRef.current?.update(component);
    }
    else componentTargets.current.delete(component.id);
  }, []);
  const componentMatchesPermalink = useCallback(
    (authoredId, permalinkId) =>
      authoredId === permalinkId ||
      (validComponentId(authoredId) &&
        (componentPermalinkId(globalThis.location, authoredId) === permalinkId ||
          componentPermalinkShortId(globalThis.location, authoredId) === permalinkId)),
    [],
  );
  const clearComponentHighlight = useCallback(() => {
    if (!componentHighlight.current) return;
    clearTimeout(componentHighlight.current.timeout);
    componentHighlight.current.element.removeAttribute("data-permalink-target");
    componentHighlight.current = null;
  }, []);
  const followComponentPermalink = useCallback(() => {
    const nextComponent = readComponentPermalink(window.location);
    clearComponentHighlight();
    handledComponentPermalink.current = null;
    unavailableComponentPermalink.current = null;
    componentTabProbe.current = null;
    chartEditorRef.current?.followPermalink(nextComponent, componentMatchesPermalink);
    setLinkedComponent(nextComponent);
  }, [clearComponentHighlight, componentMatchesPermalink]);
  const [assumptions, setAssumptions] = useState(
    savedPresentation.assumptions ?? { activationLift: 0, retentionLift: 0 },
  );
  const editingRoot = useInlineEditing(
    canEdit && mode === "edit" && !editSession?.saving,
    (id, value) => setNarrativeEdits((current) => ({ ...current, [id]: value })),
    narrativeEdits,
  );
  useDataAppImageTools(editingRoot, componentTargets, {
    surface: snapshot.surface, tabId: reportSurface ? null : activeTabId, filters,
  });
  const presentation = {
    theme: activeTheme,
    appearance,
    title: appTitle,
    hiddenBlocks: [...hidden],
    componentTitles,
    textEdits: narrativeEdits,
    chartOverrides,
    ...(verification ? { verification } : {}),
    ...(Object.keys(blockLayouts).length ? { blockLayouts } : {}),
    ...(savedPresentation.description !== undefined ? { description: savedPresentation.description } : {}),
    ...(!reportSurface
      ? {
          tabs,
          ...(savedPresentation.refreshSchedule ? { refreshSchedule: savedPresentation.refreshSchedule } : {}),
        }
      : {}),
    ...(!hosted ? { filters, assumptions } : {}),
  };
  const acknowledgePresentation = useCallback((authoritative, action) => {
    const nextVerification = authoritative.verification;
    setVerification((current) =>
      JSON.stringify(current) === JSON.stringify(nextVerification) ? current : nextVerification,
    );
    if (action) setPendingVerification((current) => (current === action ? null : current));
    if (editSessionRef.current?.saving) {
      setEditSession(null);
      setMode("view");
    }
  }, []);
  function restorePresentation(value) {
    setActiveTheme(value.theme ?? "original");
    setAppearance(normalizeAppearance(value.appearance));
    setAppTitle(value.title ?? snapshot.title);
    setHidden(new Set(value.hiddenBlocks ?? []));
    setComponentTitles(value.componentTitles ?? {});
    setNarrativeEdits(value.textEdits ?? {});
    setChartOverrides(value.chartOverrides ?? {});
    setBlockLayouts(value.blockLayouts ?? {});
    setTabs(value.tabs ?? defaultDashboardTabs);
  }
  const editHistory = usePresentationHistory(presentation, canEdit && mode === "edit", restorePresentation);
  const handlePresentationError = useCallback((message, action) => {
    if (action) setPendingVerification((current) => (current === action ? null : current));
    setActionStatus(message);
    setEditSession((current) => current ? { ...current, saving: false } : current);
  }, []);
  const saveStatus = usePresentationPersistence({
    snapshot,
    hosted,
    canEdit,
    presentation: editSession && !editSession.saving
      ? presentationBeforeEdits(presentation, editSession.baseline) : presentation,
    personalPresentation: { ...(hosted ? { filters, assumptions } : {}), ...(scopedTabs ? { tabViews } : {}) },
    initialPresentation: savedPresentation,
    initialRevision,
    verificationAction: pendingVerification,
    onAcknowledged: acknowledgePresentation,
    onError: handlePresentationError,
  });
  function beginEditing() {
    if (!canEdit || editSession || saveStatus === "saving") return;
    setEditSession({ baseline: JSON.parse(editHistoryValue(presentation)), saving: false });
    setMode("edit");
  }
  function cancelEditing() {
    if (!editSession || editSession.saving) return;
    restorePresentation(editSession.baseline);
    setEditSession(null);
    setMode("view");
  }
  function saveEditing() {
    if (!editSession || editSession.saving) return;
    // Text editors commit on blur. Flush that event before taking the value
    // used by persistence, including keyboard activation of the Save control.
    flushSync(() => document.activeElement?.blur?.());
    setEditSession((current) => current ? { ...current, saving: true } : current);
  }
  const editingChanged = editSession && editHistoryValue(presentation) !== JSON.stringify(editSession.baseline);
  useEffect(() => {
    if (editSession?.saving && !editingChanged) {
      setEditSession(null);
      setMode("view");
    }
  }, [editSession?.saving, editingChanged]);
  useEffect(() => {
    if (!editingChanged) return undefined;
    const warn = (event) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [editingChanged]);
  const pendingCodeChanges = usePendingCodeChanges();
  const resolveColor = useMemo(() => semanticColorResolver(snapshot.queries), [snapshot.queries]);
  const previousTheme = useRef(activeTheme);

  useEffect(() => {
    if (activeTheme !== "original" || previousTheme.current !== activeTheme) applyDataAppTheme(activeTheme);
    previousTheme.current = activeTheme;
  }, [activeTheme]);

  useEffect(() => {
    setDataAppAppearance(viewerAppearance || appearance);
  }, [viewerAppearance, appearance, activeTheme]);

  useEffect(() => {
    if (!tabs.some(({ id }) => id === activeTabId)) setActiveTabId(tabs[0].id);
  }, [activeTabId, tabs]);

  useEffect(() => {
    if (!tabRegistrationReady || reportSurface || typeof window === "undefined") return;
    const next = dashboardUrlWithState(window.location, scopedSnapshot, tabs, {
      tab: activeTabId,
      filters,
    });
    if (next && next.href !== window.location.href) {
      window.history.replaceState(window.history.state, "", next);
    }
  }, [activeTabId, filters, reportSurface, scopedSnapshot, tabs, tabRegistrationReady]);

  useEffect(() => {
    if (!scopedTabs || !tabRegistrationReady) return;
    const view = { filters, focus: viewFocus };
    if (!drillReturn) setTabViews(current => JSON.stringify(current[activeTabId]) === JSON.stringify(view)
      ? current : { ...current, [activeTabId]: view });
    if (typeof window !== "undefined") window.history.replaceState({ ...window.history.state,
      dataAppView: { artifactId: snapshot.id, ...viewRef.current } }, "");
  }, [activeTabId, filters, viewFocus, drillReturn, scopedTabs, tabRegistrationReady, snapshot.id]);

  useEffect(() => {
    if (reportSurface || typeof window === "undefined") return undefined;
    function restoreDashboardUrlState(event) {
      const definitions = tabDefinitionsRef.current;
      const requested = new URL(window.location.href).searchParams.get("tab");
      const tabId = dashboardTabId(definitions.length ? definitions : tabs, requested);
      const scoped = dashboardTabSnapshot(snapshot, definitions, tabId);
      const state = readDashboardUrlState(scoped, tabs, window.location);
      const stored = event.state?.dataAppView;
      const view = stored && stored.artifactId === snapshot.id && stored.tabId === tabId ? stored : null;
      const next = dashboardView(snapshot, definitions, tabId,
        view ?? tabViewsRef.current[tabId] ?? { filters: filtersRef.current }, state);
      setActiveTabId(tabId);
      replaceFilters(next.filters);
      setViewFocus(next.focus);
      setDrillReturn(view?.returnTo ?? null);
      if (view?.scrollY != null) requestAnimationFrame(() => window.scrollTo(0, view.scrollY));
    }
    window.addEventListener("popstate", restoreDashboardUrlState);
    return () => window.removeEventListener("popstate", restoreDashboardUrlState);
  }, [replaceFilters, reportSurface, snapshot, tabs]);

  const navigateDashboardTab = useCallback((tabId, nextTabs = tabs, payload = null) => {
    if (!nextTabs.some(tab => tab.id === tabId) || (tabId === activeTabRef.current && !payload)) return;
    const definitions = tabDefinitionsRef.current;
    const scoped = definitions.some(tab => Array.isArray(tab.filterIds));
    const saved = tabViewsRef.current[tabId] ?? {};
    const target = payload ? drillDashboardView(snapshot, definitions, tabId, saved, payload)
      : scoped ? dashboardView(snapshot, definitions, tabId, saved) : { filters: filtersRef.current, focus: {} };
    const origin = { ...viewRef.current, scrollY: globalThis.scrollY ?? 0 };
    const returnTo = payload ? origin : null;
    if (!reportSurface && typeof window !== "undefined") {
      window.history.replaceState({ ...window.history.state,
        dataAppView: { artifactId: snapshot.id, ...origin } }, "");
      const next = dashboardUrlWithState(window.location, dashboardTabSnapshot(snapshot, definitions, tabId), nextTabs,
        { tab: tabId, filters: target.filters });
      if (next) {
        next.pathname = canonicalDashboardPath(next.pathname);
        window.history.pushState({ ...window.history.state,
          dataAppView: { artifactId: snapshot.id, tabId, ...target, returnTo } }, "", next);
        if (hosted) followComponentPermalink();
      }
      requestAnimationFrame(() => window.scrollTo(0, 0));
    }
    replaceFilters(target.filters);
    setViewFocus(target.focus);
    setDrillReturn(returnTo);
    setActiveTabId(tabId);
  }, [followComponentPermalink, hosted, replaceFilters, reportSurface, snapshot, tabs]);
  const exploreDashboard = useCallback((tabId, payload) => navigateDashboardTab(tabId, tabs, payload), [navigateDashboardTab, tabs]);
  const returnFromExploration = useCallback(() => {
    if (drillReturn && typeof window !== "undefined") window.history.back();
  }, [drillReturn]);
  const setDashboardFocus = useCallback((focus) => {
    setViewFocus(dashboardView(snapshot, tabDefinitionsRef.current, activeTabRef.current, { focus }).focus);
  }, [snapshot]);

  useEffect(() => {
    if (!hosted || typeof window === "undefined") return undefined;

    window.addEventListener("popstate", followComponentPermalink);
    return () => {
      window.removeEventListener("popstate", followComponentPermalink);
      clearComponentHighlight();
      handledComponentPermalink.current = null;
      unavailableComponentPermalink.current = null;
      componentTabProbe.current = null;
    };
  }, [clearComponentHighlight, followComponentPermalink, hosted]);

  useEffect(() => {
    if (!tabRegistrationReady || !hosted || !linkedComponent || dashboardBusyRef.current) return;

    const routeKey = `${linkedComponent.kind}\u0000${linkedComponent.id}\u0000${linkedComponent.detail}`;
    if (handledComponentPermalink.current === routeKey) return;
    const unavailableMessage =
      linkedComponent.kind === "chart" ? "The linked chart is unavailable." : "The linked component is unavailable.";

    function showUnavailable() {
      if (unavailableComponentPermalink.current !== routeKey) {
        unavailableComponentPermalink.current = routeKey;
        setActionStatus(unavailableMessage);
      }
    }

    const hiddenTarget =
      hidden.has(linkedComponent.id) ||
      [...hidden].slice(0, 500).some((id) => componentMatchesPermalink(id, linkedComponent.id));
    if (hiddenTarget) {
      componentTabProbe.current = null;
      showUnavailable();
      return;
    }

    const existingTabIds = new Set(tabs.slice(0, 50).map(({ id }) => id));
    let probe = componentTabProbe.current;
    if (!probe || probe.routeKey !== routeKey) {
      probe = {
        routeKey,
        attemptedTabIds: new Set(),
        originalTabId: activeTabId,
        pendingTabId: null,
      };
      componentTabProbe.current = probe;
    }
    if (probe.pendingTabId && probe.pendingTabId !== activeTabId && existingTabIds.has(probe.pendingTabId)) {
      return;
    }
    probe.pendingTabId = null;

    const candidate =
      componentTargets.current.get(linkedComponent.id) ??
      [...componentTargets.current.values()]
        .slice(0, 500)
        .find(({ component }) => componentMatchesPermalink(component.id, linkedComponent.id));
    // Authored tabs may retain their DOM while hiding inactive panels. A
    // registered but invisible component still needs its owning tab selected.
    const target = isComponentPermalinkTargetVisible(candidate?.element) ? candidate : null;
    if (!target) {
      probe.attemptedTabIds.add(activeTabId);
      const dashboardTab = tabs.find(({ id }) => id === "dashboard" && !probe.attemptedTabIds.has(id));
      const nextTab = dashboardTab ?? tabs.slice(0, 50).find(({ id }) => !probe.attemptedTabIds.has(id));
      if (nextTab) {
        probe.pendingTabId = nextTab.id;
        setActiveTabId(nextTab.id);
        return;
      }
    }

    if (!target || (linkedComponent.kind === "chart" && !target.component.chart)) {
      handledComponentPermalink.current = routeKey;
      componentTabProbe.current = null;
      showUnavailable();
      if (probe.originalTabId !== activeTabId && existingTabIds.has(probe.originalTabId)) {
        setActiveTabId(probe.originalTabId);
      }
      return;
    }

    handledComponentPermalink.current = routeKey;
    componentTabProbe.current = null;
    if (unavailableComponentPermalink.current === routeKey) {
      unavailableComponentPermalink.current = null;
      setActionStatus((current) => (current === unavailableMessage ? "" : current));
    }
    const { component, element } = target;
    element.setAttribute("data-permalink-target", "true");
    element.scrollIntoView({
      block: "center",
      behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ? "auto" : "smooth",
    });
    element.focus({ preventScroll: true });
    if (linkedComponent.kind === "chart" && linkedComponent.detail && component.chart) {
      chartEditorRef.current?.open(
        {
          type: "explore",
          component,
          permalink: true,
          originalOverride: chartOverrides[component.id],
          dirty: false,
          resetVersion: 0,
          draft: component.chart,
          history: [component.chart],
          historyIndex: 0,
        },
        false,
      );
    }

    const timeout = setTimeout(() => {
      element.removeAttribute("data-permalink-target");
      if (componentHighlight.current?.element === element) componentHighlight.current = null;
    }, 2400);
    componentHighlight.current = { element, timeout };
  }, [
    activeTabId,
    chartOverrides,
    componentMatchesPermalink,
    hidden,
    hosted,
    linkedComponent,
    tabs,
    tabRegistrationReady,
    dashboardBusy,
  ]);

  const visible = (id) => !hidden.has(id);
  function updateChartState(id, changes) {
    setChartStates((current) => ({
      ...current,
      [id]: { ...(current[id] ?? {}), ...changes },
    }));
  }
  function chartProps(id) {
    return {
      chartId: id,
      resolveColor,
      visibleSeries: chartStates[id]?.visibleSeries,
      onVisibleSeriesChange: (visibleSeries) => updateChartState(id, { visibleSeries }),
      zoomRange: chartStates[id]?.zoomRange,
      onZoomChange: (zoomRange) => updateChartState(id, { zoomRange }),
    };
  }
  function requestTextEdit(target) {
    if (!canEdit || !target) return;
    if (mode !== "edit") beginEditing();
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        if (!target.isConnected) return;
        const editable = target.isContentEditable ? target : target.querySelector('[contenteditable="true"]');
        editable?.focus();
      }),
    );
  }
  async function copyComponent(format, component) {
    if (format === "link" || format === "detail-link" || format === "component-link") {
      try {
        const chartLink = format !== "component-link";
        if (!hosted || (chartLink && !component.chart)) {
          throw new Error("Only published components have shareable links.");
        }
        if (!navigator.clipboard?.writeText) throw new Error("Clipboard is unavailable in this browser.");
        const componentId = componentPermalinkShortId(globalThis.location, component.id);
        const permalink = chartLink
          ? chartPermalink(globalThis.location, componentId, {
              detail: format === "detail-link",
            })
          : componentPermalink(globalThis.location, componentId);
        const url = reportSurface
          ? permalink
          : dashboardUrlWithState(
              permalink,
              scopedSnapshot,
              tabs,
              {
                tab: activeTabId,
                filters,
              },
              { preserveExisting: false },
            ).toString();
        await navigator.clipboard.writeText(url);
        setActionStatus(`${component.title} link copied.`);
      } catch (error) {
        setActionStatus(error instanceof Error ? error.message : "Unable to copy the component link.");
      }
      return;
    }
    if (format === "image") {
      try {
        await copyChartImage(component.id, { description: component.description });
        setActionStatus(`Chart image for “${component.title}” copied.`);
      } catch (error) {
        setActionStatus(error instanceof Error ? error.message : "Unable to copy the chart image.");
      }
      return;
    }
    const text = reviewedComponentClipboard(component, queries, (queryId) =>
      component.sourceRowsByQuery?.[queryId]
        ?? (queryId === component.queryId ? component.sourceRows ?? component.displayRows : undefined)
        ?? reviewedRows(queryId));
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard is unavailable in this browser.");
      await navigator.clipboard.writeText(text);
      setActionStatus(`${format} for “${component.title}” copied.`);
    } catch (error) {
      setActionStatus(error instanceof Error ? error.message : "Unable to copy reviewed data.");
    }
  }
  function actionContext(options = {}) {
    return {
      ...options,
      surface: snapshot.surface,
      canEdit,
      snapshot,
      title: appTitle,
      presentation: {
        ...presentation,
        filters,
        assumptions,
        filterDefinitions: scopedSnapshot.filters ?? [],
      },
    };
  }
  async function runAction(action, options = {}) {
    try {
      if (action === "pdf") return printDataApp();
      if (action === "copy-link") {
        await navigator.clipboard.writeText(currentDataAppReference().sourceUrl ?? "");
        setActionStatus("Link copied.");
        return true;
      }
      const accepted = await submitDataAppAction(action, actionContext(options));
      if (!accepted) setActionStatus(`This host cannot complete that ${surfaceNoun} action.`);
      return accepted;
    } catch (error) {
      setActionStatus(
        error instanceof Error ? error.message : `${surfaceNoun === "report" ? "Report" : "Dashboard"} action failed.`,
      );
      return false;
    }
  }
  async function requestReportFollowUp(action, followUp) {
    if (snapshot.surface !== "report"
      || !["report-investigate", "report-investigate-update", "report-correct"].includes(action)
      || (!canEdit && (action !== "report-investigate" || followUp?.editorOnly))) {
      setActionStatus("You cannot change this report.");
      return false;
    }
    const current = { ...followUp,
      text: narrativeEdits[followUp?.narrativeId] ?? followUp?.text };
    return runAction(action, { followUp: current });
  }
  function reportFollowUpHref(followUp) {
    const current = { ...followUp,
      text: narrativeEdits[followUp?.narrativeId] ?? followUp?.text };
    const href = dataAppActionHref("report-investigate", actionContext({ followUp: current }));
    return { href, target: dataAppPromptTarget(href) };
  }
  function closeChartDialog(dialog) {
    if (dialog?.permalink) {
      const currentRoute = readComponentPermalink(globalThis.location);
      if (
        currentRoute?.kind === "chart" &&
        currentRoute.detail &&
        componentMatchesPermalink(dialog.component.id, currentRoute.id)
      ) {
        const url = new URL(window.location.href);
        url.pathname = new URL(chartPermalink(window.location, currentRoute.id)).pathname;
        window.history.replaceState(window.history.state, "", url);
        handledComponentPermalink.current = null;
        componentTabProbe.current = null;
        if (componentHighlight.current) {
          clearTimeout(componentHighlight.current.timeout);
          componentHighlight.current.element.removeAttribute("data-permalink-target");
          componentHighlight.current = null;
        }
        setLinkedComponent(readComponentPermalink(url));
      }
    }
    chartEditorRef.current?.close();
  }
  function saveChartDialog(spec, dialog) {
    if (!canEdit || dialog?.type !== "explore") return;
    const previous = chartOverrides[dialog.component.id] ?? dialog.component.chart;
    if (
      previous.y !== spec.y ||
      previous.series !== spec.series ||
      JSON.stringify(previous.fields ?? []) !== JSON.stringify(spec.fields ?? [])
    ) {
      updateChartState(dialog.component.id, { visibleSeries: undefined });
    }
    setChartOverrides((current) => ({ ...current, [dialog.component.id]: spec }));
    closeChartDialog(dialog);
  }
  const actions = {
    editMode: canEdit && mode === "edit",
    canEdit,
    published: hosted,
    onOpen: (type, component) => {
      const next = {
        type,
        component,
        ...(type === "explore"
          ? {
              originalOverride: chartOverrides[component.id],
              dirty: false,
              resetVersion: 0,
              draft: component.chart,
              history: [component.chart],
              historyIndex: 0,
            }
          : {}),
      };
      if (type !== "explore") {
        chartEditorRef.current?.close();
        sourceSidebarRef.current?.open(component);
        return;
      }
      sourceSidebarRef.current?.close();
      chartEditorRef.current?.open(next);
    },
    onHide: (id) => setHidden((current) => new Set([...current, id])),
    onCopy: copyComponent,
    additionalActions: (component) => component.chart && <ChartExportAction component={component} onSelect={() => {
      const bounds = componentTargets.current.get(component.id)?.element.getBoundingClientRect();
      const chart = resolveChartSpec(component.chart, chartOverrides[component.id]);
      setChartExport({
        component: { ...component, chart },
        rows: component.displayRows ?? reviewedRows(component.queryId, [chart.x, chart.series].filter(Boolean)),
        originalSize: { width: bounds?.width ?? 1000, height: bounds?.height ?? 600 },
        provenance: chartExportMetadata(queries[component.queryId], [
          ...activeFilters.filter(filter => !filter.queryIds || filter.queryIds.includes(component.queryId)),
          ...(component.scopeFilters ?? []),
        ]),
        visibleSeries: chartStates[component.id]?.visibleSeries,
        zoomRange: chartStates[component.id]?.zoomRange,
      });
    }} />,
    onRegisterComponent: registerComponent,
    titleOverrides: componentTitles,
    onTitleChange: (value, id) => setComponentTitles((current) => ({ ...current, [id]: value })),
  };
  const setBlockLayout = useCallback(
    (regionId, layout) => {
      if (!canEdit) return;
      setBlockLayouts((current) => ({ ...current, [regionId]: layout }));
    },
    [canEdit],
  );
  const blockLayoutContext = useMemo(() => ({ blockLayouts, setBlockLayout }), [blockLayouts, setBlockLayout]);
  const shellContext = useMemo(
    () => ({
      snapshot: scopedSnapshot,
      hosted,
      canEdit,
      queries,
      filters,
      setFilter,
      reviewedRows,
      reviewedPeriodRows,
      reviewedAggregatePeriodRows,
      activeFilters,
      mode,
      appTitle,
      setAppTitle,
      assumptions,
      setAssumptions,
      chartOverrides,
      chartStates,
      visible,
      updateChartState,
      chartProps,
      resolveColor,
      surfaceNoun,
      componentActions: actions,
      activeTabId,
      registerDashboardTabs,
      setDashboardBusy, exploreDashboard, returnFromExploration, setDashboardFocus, viewFocus, canReturnFromExploration: Boolean(drillReturn),
      hiddenBlockIds: hidden,
      narrativeEdits,
      reportFollowUpHref,
      requestReportFollowUp,
      requestTextEdit,
      setNarrativeEdit: (id, value) => setNarrativeEdits((current) => ({ ...current, [id]: value })),
    }),
    [
      scopedSnapshot,
      hosted,
      canEdit,
      queries,
      filters,
      setFilter,
      reviewedRows,
      reviewedPeriodRows,
      reviewedAggregatePeriodRows,
      activeFilters,
      mode,
      appTitle,
      assumptions,
      chartOverrides,
      chartStates,
      resolveColor,
      hidden,
      activeTabId,
      registerDashboardTabs,
      setDashboardBusy, exploreDashboard, returnFromExploration, setDashboardFocus, viewFocus, drillReturn,
      componentTitles,
      activeTheme,
      appearance,
      viewerAppearance,
      narrativeEdits,
    ],
  );

  return (
    <DataAppContext.Provider value={shellContext}>
      <DataAppBlockLayoutContext.Provider value={blockLayoutContext}>
        <DashboardAskProvider
          canEdit={canEdit}
          enabled={canUseDashboardAsk({ canEdit, mode })}
          explorationEnabled={mode === "view"}
          dashboardTitle={appTitle}
          onStatus={setActionStatus}
        >
          <div className="dashboard-root">
            <DataAppTopbar
              title={appTitle}
              generatedAt={snapshot.generatedAt}
              reportAsOf={snapshot.report?.asOf}
              status={snapshot.status}
              surface={snapshot.surface}
              mode={mode}
              onModeChange={(next) => { if (next === "edit") beginEditing(); }}
              onSave={saveEditing}
              onCancel={cancelEditing}
              saving={Boolean(editSession?.saving)}
              editHistory={editHistory}
              onTitleChange={setAppTitle}
              onAction={runAction}
              getActionHref={(action, options) => dataAppActionHref(action, actionContext(options))}
              onOpenThemes={() => setThemesOpen(true)}
              published={hosted}
              canEdit={canEdit}
              verification={verification}
              onVerificationChange={
                hosted && canEdit && !reportSurface
                  ? (verified) => setPendingVerification(verified ? "verify" : "remove")
                  : undefined
              }
              saveStatus={saveStatus}
              pendingCodeChanges={pendingCodeChanges}
              hiddenCount={hidden.size}
              onRestoreHidden={() => setHidden(new Set())}
              tabs={snapshot.surface === "report" || !tabRegistrationReady ? [] : tabs}
              activeTabId={activeTabId}
              onTabChange={navigateDashboardTab}
              onReorderTabs={setTabs}
            />
            <DataAppThemeDrawer
              open={themesOpen}
              activeTheme={activeTheme}
              originalTokens={originalThemeTokens}
              appearance={viewerAppearance || appearance}
              onAppearanceChange={(value) => {
                if (canEdit) {
                  setAppearance(value);
                  setViewerAppearance("");
                  writeViewerAppearance(snapshot, "");
                } else {
                  setViewerAppearance(value);
                  writeViewerAppearance(snapshot, value);
                }
              }}
              onClose={() => setThemesOpen(false)}
              onApply={(themeId) => {
                applyDataAppTheme(themeId);
                setActiveTheme(themeId);
                setThemesOpen(false);
              }}
              onPreview={applyDataAppTheme}
              onPreviewEnd={() => applyDataAppTheme(activeTheme)}
            />
            <DataAppToast message={actionStatus} onDismiss={() => setActionStatus("")} />
            {chartExport && <ChartExportDialog {...chartExport} resolveColor={resolveColor} onClose={() => setChartExport(null)} />}

            <main
              inert={editSession?.saving || undefined}
              ref={editingRoot}
              className={`page${snapshot.surface === "report" ? " report-page" : ""}`}
              data-data-app-content={snapshot.surface}
              data-dashboard-page={activeTabId}
              onDoubleClick={
                canEdit
                  ? (event) => {
                      const target = editableTextTarget(event.currentTarget, event.target);
                      if (target) requestTextEdit(target);
                    }
                  : undefined
              }
            >
              {children}

              <ChartEditorHost
                ref={chartEditorRef}
                reviewedRows={reviewedRows}
                resolveColor={resolveColor}
                chartStates={chartStates}
                canEdit={canEdit}
                onClose={closeChartDialog}
                onSave={saveChartDialog}
              />
              <SourceSidebarHost
                ref={sourceSidebarRef}
                queries={queries}
                reviewedRows={reviewedRows}
                activeFilters={activeFilters}
                chartStates={chartStates}
                snapshot={snapshot}
              />
            </main>
          </div>
        </DashboardAskProvider>
      </DataAppBlockLayoutContext.Provider>
    </DataAppContext.Provider>
  );
}
