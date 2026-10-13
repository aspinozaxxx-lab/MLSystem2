import { useCallback, useEffect, useRef, useState } from "react";

import { apiJson } from "./api/client";
import type { TestSampleClassIndexResponse } from "./api/types";
import { useRememberedClass } from "./ClassNavigation";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;

export function useTestMarkupClasses(run: Runner, includeEmpty = false) {
  const [index, setIndex] = useState<TestSampleClassIndexResponse | null>(null);
  const { classKey, setClassKey } = useRememberedClass(index?.classes ?? null);
  const revision = useRef(0);
  const loadClasses = useCallback(async () => {
    const request = ++revision.current;
    const payload = await run(() => apiJson<TestSampleClassIndexResponse>(`/test-samples/classes${includeEmpty ? "?include_empty=true" : ""}`));
    if (!payload || request !== revision.current) return;
    setIndex(payload);
  }, [run, includeEmpty]);
  useEffect(() => { void loadClasses(); return () => { revision.current += 1; }; }, [loadClasses]);
  return { index, classKey, setClassKey, loadClasses };
}
