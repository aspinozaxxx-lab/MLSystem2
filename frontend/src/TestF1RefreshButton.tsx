import { RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

import { apiJson } from "./api/client";

export function TestF1RefreshButton({ status, endpoint, label, onQueued }: {
  status: string | null | undefined;
  endpoint: string;
  label: string;
  onQueued?: () => void | Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [queued, setQueued] = useState(false);
  useEffect(() => { setQueued(false); }, [status]);
  if (status === "current") return null;
  const active = queued || status === "queued" || status === "running";
  return <span className="test-f1-refresh-control">
    <button className="secondary icon-button test-f1-refresh" type="button"
      aria-label={`Пересчитать F1: ${label}`} title={active ? "Расчёт F1 уже в очереди или выполняется" : `Пересчитать F1: ${label}`}
      disabled={busy || active} onClick={async (event) => {
        event.stopPropagation();
        setBusy(true); setError("");
        try {
          const response = await apiJson<{ status?: string; evaluation?: { status: string }; error?: string }>(endpoint, { method: "POST" });
          const nextStatus = response.status || response.evaluation?.status;
          setQueued(nextStatus === "queued" || nextStatus === "running");
          if (nextStatus === "error" || nextStatus === "unavailable") setError(response.error || "Расчёт недоступен: проверьте снимки и сеть.");
          await onQueued?.();
        } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось поставить F1 в очередь."); }
        finally { setBusy(false); }
      }}><RefreshCw size={14} className={busy || active ? "spinning" : undefined} /></button>
    {error ? <small className="error-text" role="alert">{error}</small> : null}
  </span>;
}
