import { useMemo } from "react";
import React, { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useApolloClient } from "@apollo/client";
import { useParams } from "react-router-dom";

import pathOr from "ramda/src/pathOr";
import pipeWith from "ramda/src/pipeWith";
import isNil from "ramda/src/isNil";

import { useClientRect } from "hooks/use-client-rect";
import { useGeoViewport } from "hooks/use-geo-viewport";
import { MapboxMap, MapLoader } from "components";
import { AoiGeoJsonLayer, ProcessingResultsLayer } from "containers";
import BasemapSwitcher from "components/basemap-switcher";
import { Statuses } from "constants/common";
import MapboxDrawControl from "components/mapbox-draw-control";
import DrawRectangleButton from "components/draw-rectangle-button";
import { GET_PROCESSING } from "../processing-sidebar/index";
import { RasterLayer } from "containers/raster-layer";
import { NSPDRASTERMAPURL } from "containers/aoi-geo-json-layer/constants";
import { BASEMAP_URL } from "constants/envs";

let ProcessinngMap = React.forwardRef(({ mapAPI, aoiResultData }, ref) => {
  const mapNodeRef = useRef();
  const { processingId } = useParams();

  const client = useApolloClient();
  const { data, status } = useQuery({
    queryKey: ["processing", processingId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_PROCESSING,
        fetchPolicy: "no-cache",
        variables: { processingId },
      });
      return result?.data?.processing;
    },
  });
  const [size, rectNodeRef] = useClientRect();

  const [isAoiLayerLoaded, setIsAoiLayerLoaded] = useState(false);

  const isDataDefined = status === "success";
  const bbox = pathOr(null, ["bbox"])(data);
  const vectorLayer = pathOr(null, ["vectorLayer"])(data);

  const viewport = useGeoViewport(parseJson(bbox), size);
  const isDataReady = isDataDefined && viewport;
  const isMapReady = mapAPI && isAoiLayerLoaded;

  const isGeotiffDataProvider =
    pathOr(null, ["dataProvider", "name"])(data) === "GTIFF";
  const dataProvider = pathOr(null, ["dataProvider"])(data);
  const progress = data?.progress;
  const rasterTileUrl = data?.rasterLayer?.tileUrl;
  const completedArea = useMemo(() => {
    if (!progress?.details) return 0;
    return progress.details.reduce((acc, { status, area }) => {
      if (status === Statuses.SUCCESS) return area + acc;
      return acc;
    }, 0);
  }, [progress]);

  const isSuccesAoi = aoiResultData?.some(
    (aoi) => aoi.progress.status === "OK",
  );

  return (
    <div ref={rectNodeRef} className="processing-map">
      {isDataReady && (
        <MapboxMap
          ref={{ mapNodeRef, ref }}
          viewport={
            data?.aoiCount === 0
              ? { center: [37.6173, 55.7558], zoom: 10 }
              : { ...viewport, zoom: viewport.zoom - 1 }
          }
          transformRequest={(url, resourceType) => {
            if (resourceType === "Tile" && !url.includes("api.mapbox.com")) {
              return {
                url: url,
                headers: { "Cache-Control": "max-age=0" },
              };
            }
          }}
        />
      )}

      {mapAPI && (
        <AoiGeoJsonLayer
          mapAPI={mapAPI}
          onLoaded={() => setIsAoiLayerLoaded(true)}
        />
      )}

      {mapAPI && (
        <RasterLayer
          mapAPI={mapAPI}
          layerId={NSPDRASTERMAPURL}
          url={BASEMAP_URL}
        />
      )}

      {mapAPI && rasterTileUrl && isSuccesAoi && (
        <RasterLayer
          mapAPI={mapAPI}
          layerId="wm-raster-tile-url"
          url={rasterTileUrl}
        />
      )}

      {mapAPI && vectorLayer && (
        <ProcessingResultsLayer
          mapAPI={mapAPI}
          vectorLayer={vectorLayer}
          tileUrl={data?.rasterLayer?.tileUrl}
          completedArea={completedArea}
        />
      )}
      <div
        style={{
          position: "absolute",
          top: "0",
          left: "0",
          marginTop: "2rem",
          marginLeft: "2rem",
        }}
      >
        {mapAPI && (
          <BasemapSwitcher
            dataProvider={dataProvider}
            resultsLayerId={vectorLayer}
          />
        )}
        {mapAPI && (
          <MapboxDrawControl>
            <div style={{ marginTop: "1rem" }}>
              <DrawRectangleButton
                isGeotiffDataProvider={isGeotiffDataProvider}
                mapAPI={mapAPI}
              />
            </div>
          </MapboxDrawControl>
        )}
      </div>
      <MapLoader isDataLoading={!isDataReady} isMapLoading={!isMapReady} />
    </div>
  );
});

const pipeWhileNotNil = pipeWith((f, res) => (isNil(res) ? res : f(res)));
const parseJson = pipeWhileNotNil([JSON.parse]);

export default React.memo(ProcessinngMap);
