const LAYER_BEFORE_DRAWING_GL = "gl-draw-polygon-fill-inactive.cold";

const getSourceId = (layerId) => `${layerId}-source`;

export function addRasterLayer(
  mapAPI,
  {
    url,
    type = "xyz",
    layerId = "USER_RASTER_ID",
    beforeId = LAYER_BEFORE_DRAWING_GL, // to prevent showing layer above the user drawings
  },
) {
  const sourceId = getSourceId(layerId);

  if (!mapAPI.getSource(sourceId)) {
    mapAPI.addSource(sourceId, {
      type: "raster",
      tiles: [url],
      tileSize: 256,
      scheme: type,
    });

    const newLayer = {
      id: `${layerId}`,
      type: "raster",
      source: sourceId,
      layout: { visibility: "visible" },
    };

    // If layer with this id already on
    // the map remove it and then add new one
    if (mapAPI.getLayer(newLayer.id)) mapAPI.removeLayer(newLayer.id);

    if (mapAPI.getLayer(beforeId)) mapAPI.addLayer(newLayer, beforeId);
    else mapAPI.addLayer(newLayer);
  }
}

export function removeRasterLayer(mapAPI, layerId) {
  const sourceId = getSourceId(layerId);

  if (mapAPI.getSource(sourceId)) {
    mapAPI.removeLayer(`${layerId}`);
    mapAPI.removeSource(sourceId);
  }
}
