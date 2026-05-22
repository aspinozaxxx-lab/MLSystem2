export const LayerIds = {
  PROCESSED: "wm-processed-fill-extrusion",
  PROCESSED_OUTLINE: "wm-processed-outline",
  PROCESSED_POINTS: "point",
  PROCESSED_LINES: "wm-lines",
};

export const SourceIds = {
  PROCESSED: "wm-processed",
};

const polygonFilter = ["==", "$type", "Polygon"];
const lineFilter = ["==", "$type", "LineString"];
const symbolFilter = ["==", "$type", "Point"];

export const LayerStyles = {
  getProcessed(id, source) {
    return {
      id: LayerIds.PROCESSED,
      type: "fill-extrusion",
      source,
      "source-layer": "vector_layer",
      minzoom: 9,
      filter: polygonFilter,
      paint: {
        "fill-extrusion-height": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9,

          ["get", "building_height"],
        ],
        "fill-extrusion-opacity": ["interpolate", ["linear"], ["zoom"], 9, 1],
        "fill-extrusion-color": "#ff1493",
      },
    };
  },
  getProcessedOutline(id, source) {
    return {
      id: LayerIds.PROCESSED_OUTLINE,
      type: "line",
      source,
      "source-layer": "vector_layer",
      filter: polygonFilter,
      minzoom: 9,
      paint: {
        "line-color": "#ff1493",
        "line-width": ["interpolate", ["linear"], ["zoom"], 9, 1],
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 1],
      },
    };
  },

  getProcessedLines(id, source) {
    return {
      id: LayerIds.PROCESSED_LINES,
      type: "line",
      source,
      "source-layer": "vector_layer",
      filter: lineFilter,
      minzoom: 9,
      paint: {
        "line-color": "#ff1493",
        "line-width": 10,
        "line-opacity": 1,
      },
    };
  },

  getProcessedSymbols(id, source) {
    return {
      id: "point",
      type: "circle",
      source,
      "source-layer": "vector_layer",
      filter: symbolFilter,
      minzoom: 9,
      paint: {
        "circle-radius": 5,
        "circle-color": "#ff1493",
      },
    };
  },
};
