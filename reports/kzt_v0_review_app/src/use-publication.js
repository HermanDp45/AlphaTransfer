import { useEffect, useState } from "react";

export function usePendingCodeChanges() {
  const [pending, setPending] = useState(() =>
    globalThis.__DATA_DASHBOARD_PENDING_CODE_CHANGES__ === true);

  useEffect(() => {
    const update = (event) => setPending(event.detail?.pending !== false);
    globalThis.addEventListener?.("data-app:code-changes", update);
    return () => globalThis.removeEventListener?.("data-app:code-changes", update);
  }, []);

  return pending;
}
