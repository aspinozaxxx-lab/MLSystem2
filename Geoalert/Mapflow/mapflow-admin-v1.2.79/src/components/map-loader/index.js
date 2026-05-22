import React, { useMemo } from "react";
import { t, Trans } from "@lingui/macro";

import { StateLoading } from "components";

const MAP_LOADING = "MAP_LOADING";
const DATA_LOADING = "DATA_LOADING";
const States = {
  MAP_LOADING: MAP_LOADING,
  DATA_LOADING: DATA_LOADING,
  T: {
    [MAP_LOADING]: t`Initialize the map`,
    [DATA_LOADING]: t`Fetching the processing`,
  },
};

function getCurrentState({ isDataLoading, isMapLoading }) {
  if (isDataLoading) return States.DATA_LOADING;
  if (isMapLoading) return States.MAP_LOADING;
}

function MapLoader({ isDataLoading, isMapLoading }) {
  const currentState = getCurrentState({ isDataLoading, isMapLoading });
  return useMemo(() => {
    if (currentState)
      return (
        <StateLoading
          className="map-loader-spinner"
          title={<Trans id="Loading" />}
          description={<Trans id={States.T[currentState]} />}
        />
      );
    return null;
  }, [currentState]);
}

export default React.memo(MapLoader);
