import { useCallback, useEffect, useRef, useState } from "react";

import { apiJson } from "./api/client";
import type { TestSampleClassIndexResponse } from "./api/types";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
const storageKey = "grovika:test-markups:class";

export function selectAvailableTestMarkupClass(keys: string[], current: string): string {
  return keys.includes(current) ? current : keys[0] || "";
}

export function useTestMarkupClasses(run: Runner, includeEmpty = false) {
  const [index, setIndex] = useState<TestSampleClassIndexResponse | null>(null);
  const [classKey, setClassKey] = useState(() => {
    try { return localStorage.getItem(storageKey) || ""; } catch { return ""; }
  });
  const revision = useRef(0);
  const loadClasses = useCallback(async () => {
    const request = ++revision.current;
    const payload = await run(() => apiJson<TestSampleClassIndexResponse>(`/test-samples/classes${includeEmpty ? "?include_empty=true" : ""}`));
    if (!payload || request !== revision.current) return;
    setIndex(payload);
    setClassKey((current) => selectAvailableTestMarkupClass((payload.classes || []).map((item) => item.key), current));
  }, [run, includeEmpty]);
  useEffect(() => { void loadClasses(); return () => { revision.current += 1; }; }, [loadClasses]);
  useEffect(() => {
    if (!index || !classKey) return;
    try { localStorage.setItem(storageKey, classKey); } catch { /* Выбор работает и без хранилища браузера. */ }
  }, [index, classKey]);
  return { index, classKey, setClassKey, loadClasses };
}
