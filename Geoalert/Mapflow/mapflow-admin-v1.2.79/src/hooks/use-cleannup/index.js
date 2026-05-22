import { useEffect } from "react";

export function useCleanup(value, cleanFn) {
  useEffect(() => {
    // console.log({ value, cleanFn });
    if (!value || !cleanFn) return;
    return cleanFn;
  }, [value, cleanFn]);
}
