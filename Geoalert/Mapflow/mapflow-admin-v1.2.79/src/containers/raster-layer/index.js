import { LayerIds } from "containers/processing-results-layer/constants";
import { useEffect } from "react";
import { addRasterLayer, removeRasterLayer } from "shared/geo/layers";

export const RasterLayer = ({ mapAPI, url, layerId }) => {
  useEffect(() => {
    if (!url || !mapAPI || !layerId) return;

    addRasterLayer(mapAPI, {
      url,
      layerId,
      beforeId: LayerIds.PROCESSED,
    });

    return () => {
      try {
        if (mapAPI) {
          removeRasterLayer(mapAPI, layerId);
        }
      } catch (error) {}
    };
  }, [mapAPI, url, layerId]);

  return null;
};
