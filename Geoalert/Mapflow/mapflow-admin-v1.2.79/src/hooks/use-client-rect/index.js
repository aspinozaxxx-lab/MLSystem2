import { useState, useCallback } from "react";

function getSizes(node) {
  const { width, height } = node.getBoundingClientRect();
  return [width, height];
}

export function useClientRect() {
  const [rect, setRect] = useState(null);
  const ref = useCallback((node) => {
    if (node !== null) {
      setRect(() => getSizes(node));
    }
  }, []);
  return [rect, ref];
}
