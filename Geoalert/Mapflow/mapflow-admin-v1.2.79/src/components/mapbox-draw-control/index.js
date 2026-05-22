import React, {
  useMemo,
  useState,
  useEffect,
  useContext,
  useCallback,
  createContext,
} from "react";
import { useMapboxGlDraw, defaultDrawOptions } from "hooks/use-mapbox-gl-draw";

import DrawRectangle from "mapbox-gl-draw-rectangle-restrict-area";
import MapboxDraw from "@mapbox/mapbox-gl-draw";

export type DrawFnOptions = {
  areaLimit?: number,
  onExceed?: (e: any) => void,
  onCancel?: (e: any) => void,
  areaChanged?: (area: number) => void,
};

const Context = createContext({
  drawAPI: null,
  drawRectangle: () => {},
});

export function useDrawAPI() {
  return useContext(Context);
}

type Handler = (e: any, drawAPI: any, mapAPI: any) => void;

// const previousModeRef = useRef("simple_select");
const MapboxDrawControl: React.FC<{
  onDrawUpdated?: Handler,
  onDrawCreated?: Handler,
  onModeChange?: Handler,
  onDrawCancel?: Handler,
}> = ({
  children,
  onDrawCreated,
  onModeChange,
  onDrawUpdated,
  onDrawCancel,
}) => {
  const [drawAPI, setDrawAPI] = useState(null);

  useMapboxGlDraw(
    window.mapboxMap,
    {
      ...defaultDrawOptions,
      modes: Object.assign(MapboxDraw.modes, {
        draw_rectangle: DrawRectangle,
      }),
    },
    setDrawAPI,
  );

  const handleMapboxEvent = (fn: Handler) => (e: any) =>
    fn && fn(e, drawAPI, window.mapboxMap);

  useEffect(() => {
    if (!window.mapboxMap) return;
    if (!drawAPI) return;

    const map = window.mapboxMap;

    map.on("draw.create", handleMapboxEvent(onDrawCreated));
    map.on("draw.modechange", handleMapboxEvent(onModeChange));
    map.on("draw.update", handleMapboxEvent(onDrawUpdated));
  }, [drawAPI]); // eslint-disable-line react-hooks/exhaustive-deps

  const drawRectangle = useCallback(
    ({ areaLimit, onExceed, areaChanged, onCancel }: DrawFnOptions = {}) => {
      if (!drawAPI) return;

      // 10 km2, optional
      drawAPI.changeMode("draw_rectangle", {
        areaLimit: areaLimit || 10_000_000,
        exceedCallback: onExceed,
        areaChangedCallback: areaChanged,
      });

      window.mapboxMap.once("draw.modechange", (e) => {
        window.mapboxMap.getCanvas().style.cursor = "default";

        const isFeatures = drawAPI.getAll().features.length > 0;
        if (e.mode === "simple_select" && !isFeatures) {
          handleMapboxEvent(onDrawCancel)(e);
          onCancel && onCancel(e);
        }
      });
      window.mapboxMap.getCanvas().style.cursor = "crosshair";

      return drawAPI;
    },
    [drawAPI], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const context = useMemo(() => ({ drawAPI, drawRectangle }), [
    drawAPI,
    drawRectangle,
  ]);

  return <Context.Provider value={context}>{children}</Context.Provider>;
};

export default MapboxDrawControl;
