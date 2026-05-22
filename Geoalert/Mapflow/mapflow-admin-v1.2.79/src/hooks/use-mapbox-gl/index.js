/* globals mapboxgl */
import { useEffect } from "react";
import styles from "./styles.json";
import { avanpostCookies } from "providers/auth-avanpost-provider";

export const defaultMapOptions = {
  style: styles,
  center: [0, 0],
  zoom: 0,
};

export function useMapboxGl(nodeRef, setMapAPI, options) {
  useEffect(() => {
    const container = nodeRef.current;
    if (!container) return;
    const tileUrlPrefix = process.env.REACT_APP_TILE_URL_PREFIX;
    const mapOptions = {
      container,
      ...defaultMapOptions,
      ...options,
      transformRequest: (url) => {
        if (url.startsWith(tileUrlPrefix)) {
          const { token } = avanpostCookies.getTokens();
          return {
            url,
            headers: { Authorization: `Bearer ${token}` },
          };
        }
      },
    };
    const mapboxGlMap = new mapboxgl.Map(mapOptions);
    mapboxGlMap.once("load", () => setMapAPI(mapboxGlMap));
    window.mapboxMap = mapboxGlMap;
    return () => {
      mapboxGlMap.remove();
      delete window.mapboxMap;
    };
  }, [nodeRef]); // eslint-disable-line react-hooks/exhaustive-deps
}
