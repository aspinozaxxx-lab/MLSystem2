import { useEffect, useRef } from "react";

import { apiJson } from "./api/client";
import type { UsageConfig } from "./api/types";
import { startMetrica, usagePageForRoute, usagePages, type UsagePage, type UsageTracker } from "./utils/usage";

export function useUsage(username: string | null, route: readonly string[]): void {
  const tracker = useRef<UsageTracker | null>(null);
  const currentPage = useRef<UsagePage>(usagePageForRoute(route));
  const routeKey = route.join("/");
  const previousRoute = useRef(routeKey);

  useEffect(() => {
    if (previousRoute.current !== routeKey) {
      previousRoute.current = routeKey;
      currentPage.current = usagePageForRoute(routeKey.split("/"));
      tracker.current?.page(currentPage.current);
    }
  }, [routeKey]);

  useEffect(() => {
    if (!username) return;
    let cancelled = false;
    currentPage.current = usagePageForRoute(previousRoute.current.split("/"));
    const tabPage = (event: Event) => {
      const page = (event as CustomEvent<UsagePage>).detail;
      if (!Object.hasOwn(usagePages, page) || currentPage.current === page) return;
      currentPage.current = page;
      tracker.current?.page(page);
    };
    const action = (event: Event) => {
      const payload = (event as CustomEvent<{ name: string; status: number; userId: string }>).detail;
      tracker.current?.action(payload.name, payload.status, payload.userId);
    };
    window.addEventListener("grovika:page", tabPage);
    window.addEventListener("grovika:action", action);
    void apiJson<UsageConfig>("/usage/config").then((config) => {
      if (cancelled) return;
      tracker.current = startMetrica(config);
      tracker.current?.page(currentPage.current);
    }).catch(() => { /* Счётчик не мешает работе приложения. */ });
    return () => {
      cancelled = true;
      window.removeEventListener("grovika:page", tabPage);
      window.removeEventListener("grovika:action", action);
      tracker.current?.stop();
      tracker.current = null;
    };
  }, [username]);
}
