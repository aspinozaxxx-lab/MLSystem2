import React, { useCallback } from "react";
import { gql, useApolloClient, useLazyQuery } from "@apollo/client";
import { Trans, t } from "@lingui/macro";
import { IconNames } from "@blueprintjs/icons";
import { Intent, Menu, MenuItem } from "@blueprintjs/core";

import { useMutationWithToasts } from "../../hooks/use-mutation-with-toasts";
import { getDownloadLink, invokeDownload } from "./download-by-url";
import { useErrorToast } from "hooks/use-error-toast";
import { useDownloadAsFile } from "hooks/use-download-as-file";
import { feature, featureCollection } from "@turf/helpers";
import { useQuery } from "@tanstack/react-query";
import { GET_PROCESSING } from "containers/processing-sidebar";
import { useParams } from "react-router-dom";
import { useDownload } from "hooks/useRestDownload";

export const RASTER_URL = gql`
  query rasterUrl($id: ID!) {
    rasterUrl(id: $id)
  }
`;

export const GET_AOIS_GEOMETRIES = gql`
  query getAois($filter: AoiFilter!) {
    aois(filter: $filter) {
      aois {
        id
        geometry
      }
    }
  }
`;

export const RESTART_PROCESSING = gql`
  mutation restartProcessing($id: ID!) {
    restartProcessing(processingId: $id)
  }
`;

export const CANCEL_PROCESSING = gql`
  mutation cancelProcessing($id: ID!) {
    cancelProcessing(id: $id)
  }
`;

export const DELETE_AOIS = gql`
  mutation deleteAois($filter: AoiFilter!) {
    deleteAois(filter: $filter)
  }
`;

export const GET_PROCESSING_RESULT = gql`
  mutation createProcessingResult($filter: AoiFilter!) {
    createResult(filter: $filter)
  }
`;

let ActionsMenu = ({
  filter = {},
  onUpload,
  isGeotiffDataProvider,
  isFailed,
  isPending,
  processingCommand,
  updateProcessingCommand,
  setProcessingCommand,
}) => {
  const { processingId } = useParams();
  const client = useApolloClient();

  const processingInfo = useQuery({
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

  const rasterUrl = useQuery({
    queryKey: ["rasterUrl", processingId],
    queryFn: async () => {
      const result = await client.query({
        query: RASTER_URL,
        fetchPolicy: "no-cache",
        variables: { id: processingId },
      });
      return result?.data?.rasterUrl;
    },
  });

  const downloadByRasterUrl = useCallback(() => {
    const link = document.createElement("a");
    link.href = rasterUrl.data;
    link.download = processingInfo.data.rasterLayer.id;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [processingInfo, rasterUrl]);

  const [removeAois, removeAoisResult] = useMutationWithToasts(
    DELETE_AOIS,
    {
      options: {
        variables: { filter },
        refetchQueries: ["getAois", "processing"],
      },
      getSuccesMessage: ({ data }) => t`Deleted ${data.deleteAois} AOIs`,
      getDefaultErrorMessage: () => t`Error delete AOIs`,
      pendingIntent: Intent.DANGER,
      pendingIcon: IconNames.TRASH,
    },
    updateProcessingCommand,
  );

  const [restartProcessing, restartProcessingResult] = useMutationWithToasts(
    RESTART_PROCESSING,
    {
      options: {
        variables: { id: processingId },
        refetchQueries: ["getAois", "processing"],
      },
      getSuccesMessage: ({ data }) => t`Restarted processing`,
      getDefaultErrorMessage: () => t`Error restart processing`,
      pendingIntent: Intent.PRIMARY,
      pendingIcon: IconNames.RESET,
    },
    updateProcessingCommand,
  );

  const [cancelProcessing, cancelProcessingResult] = useMutationWithToasts(
    CANCEL_PROCESSING,
    {
      options: {
        variables: { id: processingId },
        refetchQueries: ["getAois", "processing"],
      },
      getSuccesMessage: ({ data }) => t`Canceled processing`,
      getDefaultErrorMessage: () => t`Error cancel processing`,
      pendingIntent: Intent.PRIMARY,
      pendingIcon: IconNames.RESET,
    },
    updateProcessingCommand,
  );

  const { downloadFile, loading } = useDownload(`${processingId}/result`);

  const handleDownloadClick = async () => {
    setProcessingCommand(true);
    await downloadFile();         
    setProcessingCommand(false);
  };

  const onDownloadAoiGeometryError = useErrorToast({
    message: t`Error export AOI geometry`,
  });
  const download = useDownloadAsFile({ onError: onDownloadAoiGeometryError });
  const downloadAoiGeoJsons = useCallback(
    ({ aois: { aois } }) => {
      const data = featureCollection(
        aois.map((aoi) => feature(JSON.parse(aoi.geometry))),
      );
      download(
        data,
        `AreaOfInterestsGeometries_${filter.processingIds.join("_")}.json`,
      );
      updateProcessingCommand();
    },
    [download, filter.processingIds, updateProcessingCommand],
  );
  const [downloadAoiGeometry, downloadAoiGeometryResult] = useLazyQuery(
    GET_AOIS_GEOMETRIES,
    {
      fetchPolicy: "network-only",
      variables: { filter },
      onCompleted: downloadAoiGeoJsons,
      onError: onDownloadAoiGeometryError,
    },
  );

  return (
    <Menu large>
      {processingInfo.data.sourceType === "local" && (
        <MenuItem
          intent={Intent.PRIMARY}
          icon={IconNames.DOWNLOAD}
          text={<Trans id="Download raster" />}
          onClick={downloadByRasterUrl}
          disabled={processingInfo.isLoading || rasterUrl.isLoading}
        />
      )}
      <MenuItem
        intent={Intent.NONE}
        icon={IconNames.DOWNLOAD}
        text={<Trans>Download AOIs geometries</Trans>}
        onClick={() => {
          setProcessingCommand(true);
          downloadAoiGeometry();
        }}
        disabled={downloadAoiGeometryResult.loading || processingCommand}
      />
      <MenuItem
        intent={Intent.SUCCESS}
        icon={IconNames.UPLOAD}
        text={<Trans>Upload Areas of interest</Trans>}
        onClick={onUpload}
        disabled={isGeotiffDataProvider}
      />
      <MenuItem
        intent={Intent.WARNING}
        icon={IconNames.DOWNLOAD}
        text={<Trans>Export results</Trans>}
        onClick={() => {          
          handleDownloadClick();
        }}
        disabled={loading || processingCommand}
      />

      {isPending && (
        <MenuItem
          intent={Intent.PRIMARY}
          icon={IconNames.PAUSE}
          text={<Trans>Cancel processing</Trans>}
          onClick={() => {
            setProcessingCommand(true);
            cancelProcessing();
          }}
          disabled={cancelProcessingResult.loading || processingCommand}
        />
      )}

      {isFailed && (
        <MenuItem
          intent={Intent.PRIMARY}
          icon={IconNames.RESET}
          text={<Trans>Restart processing</Trans>}
          onClick={() => {
            setProcessingCommand(true);
            restartProcessing();
          }}
          disabled={restartProcessingResult.loading || processingCommand}
        />
      )}
      <MenuItem
        intent={Intent.DANGER}
        icon={IconNames.TRASH}
        text={<Trans>Delete all</Trans>}
        onClick={() => {
          setProcessingCommand(true);
          removeAois();
        }}
        disabled={removeAoisResult.loading || processingCommand}
      />
    </Menu>
  );
};

ActionsMenu = React.memo(ActionsMenu);

export const onGetExportIdSuccess = function (data) {
  if (!data) return;
  const resultId = data.createResult;
  const downloadLink = getDownloadLink(resultId);
  invokeDownload(downloadLink, `${resultId}.geojson`);
};

export { ActionsMenu };
