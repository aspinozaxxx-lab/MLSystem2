export const COLOR_BY_STATE = [
  "match",
  ["get", "status"],
  "UNPROCESSED",
  "#8A9BA8",
  "OK",
  "#0F9960",
  "IN_PROGRESS",
  "#2965CC",
  "FAILED",
  "#D13913",
  "CANCELLED",
  "#088",
  "#088",
];

export const NSPDRASTERMAPURL = "nspd-raster-map-url";
export const AoiSourcesIds = { AOI_GEOJSON: "wm-aoi-geojson" };
export const AoiLayerIds = {
  AOI_POLYGONS: "wm-aoi-polygons",
  AOI_POINTS: "wm-aoi-points",
  AOI_OUTLINE: "wm-aoi-outline",
};

export const AoiSources = {
  AOI_GEOJSON: {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  },
};

export const AoiLayers = {
  AOI_POLYGONS: {
    id: AoiLayerIds.AOI_POLYGONS,
    type: "fill",
    source: AoiSourcesIds.AOI_GEOJSON,
    filter: ["==", "$type", "Polygon"],
    paint: {
      "fill-color": COLOR_BY_STATE,
      "fill-opacity": [
        "interpolate",
        ["linear"],
        ["zoom"],
        1,
        1,
        2,
        0.9,
        3,
        0.8,
        4,
        0.7,
        5,
        0.6,
        6,
        0.5,
        7,
        0.4,
        8,
        0.3,
        9,
        0.2,
        10,
        0.1,
      ],
    },
  },
  AOI_POINTS: {
    id: AoiLayerIds.AOI_POINTS,
    type: "circle",
    source: AoiSourcesIds.AOI_GEOJSON,
    filter: ["==", "$type", "Point"],
    layout: {
      "circle-sort-key": ["/", 1, ["get", "radius"]],
    },
    paint: {
      "circle-opacity": 0.8,
      "circle-color": COLOR_BY_STATE,
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        0,
        ["*", 1.2, ["get", "radius"]],
        10,
        ["*", 1, ["get", "radius"]],
        20,
        ["*", 1, ["get", "radius"]],
      ],
    },
  },
  AOI_OUTLINE: {
    id: AoiLayerIds.AOI_OUTLINE,
    type: "line",
    source: AoiSourcesIds.AOI_GEOJSON,
    minzoom: 9,
    paint: {
      "line-color": COLOR_BY_STATE,
      "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1, 9, 3],
      "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 0.3],
    },
  },
};
