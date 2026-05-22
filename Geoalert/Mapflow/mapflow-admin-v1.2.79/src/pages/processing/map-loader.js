import React, { useMemo } from "react";
import { t, Trans } from "@lingui/react";

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

let MapLoader = function (props) {
  const currentState = getCurrentState(props);
  return useMemo(() => {
    if (currentState)
      return (
        <StateLoading
          className="processing-map-spinner"
          title={<Trans id="Loading" />}
          description={<Trans id={States.T[currentState]} />}
        />
      );
    return null;
  }, [currentState]);
};

MapLoader = React.memo(MapLoader);
export { MapLoader };
