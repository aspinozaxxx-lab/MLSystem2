import { useState } from "react";

export function useRefetch(refetchFn, { onError, cooldown = 0 } = {}) {
  const [refetching, setRefetching] = useState(false);
  const refetch = async () => {
    setRefetching(true);
    await new Promise((resolve) => setTimeout(resolve, cooldown));
    try {
      await refetchFn();
    } catch (e) {
      if (onError) onError(e);
    } finally {
      setRefetching(false);
    }
  };
  return [refetch, refetching];
}
