import { createAsyncTaksCounter } from "./create-async-task-counter";
import { useRef, useState, useEffect, useLayoutEffect } from "react";

export function useDelay(
  isLoading = false,
  { flashTreshold = 300, minLoadingTime = 700 } = {},
) {
  const [loading, setLoading] = useState(false);

  const [counter] = useState(() =>
    createAsyncTaksCounter({
      flashTreshold,
      minLoadingTime,
      show: () => setLoading(true),
      hide: () => setLoading(false),
    }),
  );

  useEffect(() => {
    if (!counter) return;

    if (isLoading) counter.startTask();
    else counter.completeTask();
  }, [isLoading, counter]);

  useEffect(() => {
    if (!counter) return;
    return () => counter.clear();
  }, [counter]);

  return loading;
}

export function _useDelay(
  isLoading = false,
  { flashTreshold = 300, minLoadingTime = 700 } = {},
) {
  const [loading, setLoading] = useState(isLoading);

  const counterRef = useRef(
    createAsyncTaksCounter({
      show: () => setLoading(true),
      hide: () => setLoading(false),
      flashTreshold,
      minLoadingTime,
      onTasksCountChange: (count) => console.log(`tasks count: ${count}`),
    }),
  );

  useLayoutEffect(() => {
    const counter = counterRef.current;
    if (!counter) return;

    if (isLoading) counter.startTask();
    else counter.completeTask();
  }, [isLoading]);

  useLayoutEffect(() => {
    const counter = counterRef.current;
    if (!counter) return;
    return () => counter.clear();
  }, []);

  return loading;
}
