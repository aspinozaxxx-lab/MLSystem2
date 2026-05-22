import { useEffect } from "react";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import { DrawStyles } from "mapbox-gl-draw-rectangle-restrict-area";

type DrawOptions = {
  userProperties: boolean,
  displayControlsDefault: boolean,
  styles: any,
};

export const defaultDrawOptions: DrawOptions = {
  userProperties: true,
  displayControlsDefault: false,
  styles: DrawStyles,
};

export const useMapboxGlDraw = (
  mapApi: null | undefined | mapboxgl.Map,
  options?: DrawOptions,
  setDrawAPI?: (drawAPI: any) => void,
) => {
  useEffect(() => {
    if (!mapApi) return;

    const draw = new MapboxDraw(options);

    window.mapboxDraw = draw;

    if (setDrawAPI) setDrawAPI(draw);

    mapApi.addControl(draw);

    return () => {
      delete window.mapboxDraw;
    };
  }, [mapApi]); // eslint-disable-line react-hooks/exhaustive-deps
};
