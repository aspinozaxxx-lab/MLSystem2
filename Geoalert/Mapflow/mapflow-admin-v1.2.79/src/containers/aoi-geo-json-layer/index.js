import React, { useRef, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import flatten from "ramda/src/flatten";

import { useQuery } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client"; 

import { AoiSources, AoiSourcesIds, AoiLayers } from "./constants";

import { POLL_INTERVAL } from "constants/envs";

const GET_AOI_GEOJSON_LAYER = gql`
  query aoiLayer($processingId: ID!, $bbox: String!, $xRes: Int!, $yRes: Int!) {
    aoiLayer(processingId: $processingId, bbox: $bbox, xRes: $xRes, yRes: $yRes)
  }
`;

const getAoiLayer = (client) => async (variables) => {
  const result = await client.query({
    query: GET_AOI_GEOJSON_LAYER, 
    fetchPolicy: "no-cache",
    variables: variables,
  });

  return result?.data?.aoiLayer;
}

function AoiGeoJsonLayer({ mapAPI, onLoaded }) {
  const { processingId } = useParams();

  const mapAPIRef = useRef(mapAPI);
  const onLoadedRef = useRef(onLoaded);

  const [ paused, setPaused ] = useState(false);

  const client = useApolloClient();

  useQuery({
    queryKey: ["processing", processingId, "aoiLayers"],
    queryFn: () => getAoiLayer(client)(getQueryVariables(processingId, mapAPIRef.current)),
    onSuccess: (data) => {
      //Create AOIs at first time load to unblock map after the data was loaded
      if (!mapAPIRef.current.getSource(AoiSourcesIds.AOI_GEOJSON)) {
        loadLayers(mapAPIRef.current);
        onLoadedRef.current();
      }
      
      mapAPIRef.current
        .getSource(AoiSourcesIds.AOI_GEOJSON)
        .setData(JSON.parse(data));
    },
    onError: (error) => console.error("Unable to fetch AOIs", error),
    refetchInterval: POLL_INTERVAL,
    enabled: !paused,
    staleTime: 0, //Disable caching becuse it depends on bbox
  });

  useEffect(() => {
    mapAPIRef.current.on("movestart", () => setPaused(true));
    mapAPIRef.current.on("moveend", () => setPaused(false));
  }, [processingId]);

  return null;
}

export default React.memo(AoiGeoJsonLayer);

function getBbox(mapAPI) {
  return JSON.stringify(flatten(mapAPI.getBounds().toArray()));
}

export function loadLayers(mapAPI) {
  mapAPI.addSource(AoiSourcesIds.AOI_GEOJSON, AoiSources.AOI_GEOJSON);
  for (const layerId in AoiLayers) {
    mapAPI.addLayer(AoiLayers[layerId]);
  }
}

function getMapSize(mapAPI) {
  const { clientWidth, clientHeight } = mapAPI.getCanvas();
  return { clientWidth, clientHeight };
}

const getQueryVariables = (processingId, mapAPI) => {
  const { clientWidth, clientHeight } = getMapSize(mapAPI);
  return {
    processingId,
    bbox: getBbox(mapAPI),
    xRes: clientWidth,
    yRes: clientHeight,
  };
};
