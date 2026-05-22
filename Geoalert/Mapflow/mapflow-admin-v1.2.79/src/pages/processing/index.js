import React, { useState, useCallback } from "react";
import { ResizeSensor } from "@blueprintjs/core";

import { MapAPIContext } from "./map-api-context";
import {
  ProcessingMap,
  ProcessingSidebar,
  UploadAoiDialog,
  Breadcrumbs,
} from "containers";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";
import { useParams } from "react-router-dom";
import { POLL_INTERVAL } from "constants/envs";

function Processing() {
  const [mapAPI, setMapAPI] = useState(null);
  const { processingId } = useParams();

  const handleResize = useCallback(() => {
    if (mapAPI) mapAPI.resize();
  }, [mapAPI]);

  const [showUploadAoiDialog, setShowUploadAoiDialog] = useState(false);

  const client = useApolloClient();

  const queryClient = useQueryClient();

  const sortQuery = useQuery({
    queryKey: ["aoiListSort"],
    queryFn: () => [],
  });

  const sort = sortQuery.data;

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["processing", processingId, "aois", sort],
    queryFn: () => getAoisList(client)({ processingId, sort }),
    refetchInterval: POLL_INTERVAL,
    onSuccess: (data) => {
      data.forEach((aoi) => {
        queryClient.setQueryData(["aoi", aoi.id], aoi);
      });
    },
  });

  const aoiResult = { data, isLoading, isError, error };

  return (
    <MapAPIContext.Provider value={mapAPI}>
      <div className="processing-page">
        <ResizeSensor onResize={handleResize}>
          <ProcessingMap
            ref={setMapAPI}
            mapAPI={mapAPI}
            aoiResultData={aoiResult.data}
          />
        </ResizeSensor>
        <Breadcrumbs className="processing-breadcrumbs" />
        <ProcessingSidebar
          aoiResult={aoiResult}
          mapAPI={mapAPI}
          openUploadAoiDialog={() => setShowUploadAoiDialog(true)}
        />
        <UploadAoiDialog
          mapAPI={mapAPI}
          isOpen={showUploadAoiDialog}
          openDialog={() => setShowUploadAoiDialog(true)}
          handleClose={() => setShowUploadAoiDialog(false)}
        />
      </div>
    </MapAPIContext.Provider>
  );
}

export default React.memo(Processing);

export const GET_AOIS = gql`
  query getAois($processingId: ID!, $sort: [AoiSortEntry!]) {
    aois(sort: $sort, filter: { processingIds: [$processingId] }) {
      aois {
        id
        area
        progress {
          status
          percentCompleted
        }
        messages {
          message
        }
      }
    }
  }
`;

const getAoisList = (client) => async (data) => {
  const result = await client.query({
    query: GET_AOIS,
    variables: data,
    fetchPolicy: "no-cache",
  });
  return result?.data?.aois?.aois;
};
