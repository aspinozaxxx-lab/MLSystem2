import geoViewport from "@mapbox/geo-viewport";
import { useMemo } from "react";

export function useGeoViewport(bbox, size) {
  return useMemo(() => {
    if (bbox && size) return geoViewport.viewport(bbox, size);
    return null;
  }, [bbox, size]);
}
