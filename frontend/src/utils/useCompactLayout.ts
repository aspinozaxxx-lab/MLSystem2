import { useSyncExternalStore } from "react";

export const COMPACT_LAYOUT_QUERY = "(max-width: 700px), (max-width: 1000px) and (max-height: 500px)";

export function isCompactLayout() {
  return typeof window !== "undefined" && window.matchMedia(COMPACT_LAYOUT_QUERY).matches;
}

function subscribe(onChange: () => void) {
  const media = window.matchMedia(COMPACT_LAYOUT_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}

// Одна граница компактного режима для CSS, списков снимков и безопасного режима карты.
export function useCompactLayout() {
  return useSyncExternalStore(subscribe, isCompactLayout, () => false);
}
