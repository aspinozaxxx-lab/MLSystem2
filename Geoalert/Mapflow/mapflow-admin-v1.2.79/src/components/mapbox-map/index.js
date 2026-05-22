import React from "react";

import { MAPBOX_TOKEN } from "constants/envs";
import { useMapboxGl } from "hooks/use-mapbox-gl";

let MapboxMap = React.forwardRef(({ viewport, ...rest }, refs) => {
  const { mapNodeRef, ref } = refs;
  useMapboxGl(mapNodeRef, ref, {
    ...rest,
    ...viewport,
    accessToken: ,
  });
  return <div id="mapbox-map" ref={mapNodeRef} />;
});

export default React.memo(MapboxMap);
