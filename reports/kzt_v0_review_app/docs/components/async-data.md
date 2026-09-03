# Async data

## Loading and permalink discovery

An asynchronous dashboard can call `setDashboardBusy(boolean)` from a layout effect and clear it on completion/unmount, pausing component-permalink discovery until real blocks register. This does not change filters or permissions.

Use `DataComponent`/`MetricCard` with `loading`, optional `loadingKind` (`chart`, `table`, `metric`, `metric-trend`), and `loadingError`. Affected cards preserve their settled frame, body height, and header, show neutral skeletons matched to chart family/category counts, and disable stale source/export actions. `loadingHeight` sets initial body geometry when it cannot be inferred from a direct chart child.

## Errors, empty results, and retained state

`onRetry` adds a recovery button only when the caller actually supports retry; errors otherwise stay informative and non-actionable. Empty charts show a clear no-data state instead of axes without marks. Keep loading announcements accessible, not additional visible labels.

Retain unaffected results only for the same snapshot and global scope; global population/date/grain changes must not display old numbers. A prior result may supply layout only during global loading.
