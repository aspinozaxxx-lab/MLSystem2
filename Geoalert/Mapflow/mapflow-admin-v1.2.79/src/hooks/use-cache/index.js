import { useState, useEffect } from "react";
import { QueryCache } from "@tanstack/react-query";

export function useCache(key) {
  const [data, setData] = useState(null);

  useEffect(() => {
    const callback = (cache) => {
      setData(cache.getQueryData(key));
    };

    const unsubscribe = QueryCache.subscribe(callback);

    return unsubscribe;
  }, [key]);

  return data;
}
