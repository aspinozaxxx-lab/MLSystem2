import React, { useMemo, useCallback, useContext } from "react";
import { Trans, t } from "@lingui/macro";
import { IconNames } from "@blueprintjs/icons";
import { Intent, Menu, MenuItem, MenuDivider } from "@blueprintjs/core";
import { feature, featureCollection } from "@turf/helpers";

import { useMutationWithToasts } from "hooks/use-mutation-with-toasts";
import { useParams } from "react-router-dom";
import { gql, useLazyQuery } from "@apollo/client";
import { useDownloadAsFile } from "hooks/use-download-as-file";
import { useErrorToast } from "hooks/use-error-toast";
import { MapAPIContext } from "pages/processing/map-api-context";
import {
  GET_PROCESSING_RESULT,
  onGetExportIdSuccess,
} from "containers/aoi-list-actions/actions-menu";
import { Statuses } from "constants/common";
import { getBounds } from "utils/bounds-from-geometry";
import { useDownload } from "hooks/useRestDownload";


export const GET_AOI = gql`
  query getAoi($id: ID!) {
    aoi(id: $id) {
      geometry
    }
  }
`;

const DELETE_AOIS = gql`
  mutation deleteAois($filter: AoiFilter!) {
    deleteAois(filter: $filter)
  }
`;

export const RUN_AOI = gql`
  mutation runAoi($id: ID!) {
    runAoi(aoiId: $id)
  }
`;

export const RESTART_AOI = gql`
  mutation restartAoi($id: ID!) {
    restartAoi(aoiId: $id)
  }
`;

let ActionsMenu = ({ aoiId, aoiStatus, isGeotiffDataProvider }) => {
  const { processingId } = useParams();

  const variables = useMemo(
    () => ({
      filter: { processingIds: [processingId], ids: [aoiId] },
    }),
    [processingId, aoiId],
  );

  const { downloadFile, loading } = useDownload(
    `${processingId}/result?aoiId=${aoiId}`,
  );

  const handleDownloadClick = async () => {
    await downloadFile();
  };

  const [removeAois, removeAoisResult] = useMutationWithToasts(DELETE_AOIS, {
    options: { variables, refetchQueries: ["getAois", "processing"] },
    getSuccesMessage: ({ data }) => t`Deleted AOI`,
    getDefaultErrorMessage: () => t`Error delete AOI`,
    pendingIntent: Intent.DANGER,
    pendingIcon: IconNames.TRASH,
  });

  const [runAoi, runAoiResult] = useMutationWithToasts(RUN_AOI, {
    options: {
      variables: { id: aoiId },
      refetchQueries: ["getAois"],
    },
    getSuccesMessage: ({ data }) => t`Successfully started AOI`,
    getDefaultErrorMessage: () => t`Error run AOI`,
    pendingIntent: Intent.PRIMARY,
    pendingIcon: IconNames.PREDICTIVE_ANALYSIS,
  });

  const onExportAoiGeometryError = useErrorToast({
    message: t`Error export AOI geometry`,
  });
  const download = useDownloadAsFile({ onError: onExportAoiGeometryError });
  const downloadAoiGeojson = useCallback(
    ({ aoi }) => {
      const data = featureCollection([feature(JSON.parse(aoi.geometry))]);
      download(data, `AreaOfInterest_${aoiId}.json`);
    },
    [aoiId, download],
  );
  const [exportAoiGeometry, getAoiResult] = useLazyQuery(GET_AOI, {
    fetchPolicy: "network-only",
    variables: { id: aoiId },
    onCompleted: downloadAoiGeojson,
    onError: onExportAoiGeometryError,
  });

  const onLocateAoiError = useErrorToast({
    message: t`Error locate AOI`,
  });
  const mapAPI = useContext(MapAPIContext);
  const fitAoiBounds = ({ aoi }) => {
    try {
      if (mapAPI) mapAPI.fitBounds(getBounds(aoi.geometry), { padding: 50 });
    } catch (error) {
      console.error(error);
      onLocateAoiError(error);
    }
  };
  const [locateAoi, locateAoiResult] = useLazyQuery(GET_AOI, {
    fetchPolicy: "network-only",
    variables: { id: aoiId },
    onCompleted: fitAoiBounds,
    onError: onLocateAoiError,
  });  return (
    <Menu>
      <MenuDivider title={<Trans>AOI actions</Trans>} />
      <MenuItem
        intent={Intent.PRIMARY}
        icon={IconNames.PREDICTIVE_ANALYSIS}
        text={<Trans id="Start AOI" />}
        onClick={runAoi}
        disabled={runAoiResult.loading || aoiStatus !== Statuses.UNPROCESSED}
      />
      <MenuItem
        intent={Intent.WARNING}
        icon={IconNames.DOWNLOAD}
        text={<Trans id="Export results" />}
        onClick={handleDownloadClick}
        disabled={loading}
      />
      <MenuItem
        intent={Intent.DANGER}
        icon={IconNames.TRASH}
        text={<Trans id="Delete" />}
        onClick={removeAois}
        disabled={removeAoisResult.loading || isGeotiffDataProvider}
      />
      <MenuDivider title={<Trans id="Map actions" />} />
      <MenuItem
        intent={Intent.NONE}
        icon={IconNames.LOCATE}
        text={<Trans id="Fit AOI bounds" />}
        onClick={locateAoi}
        disabled={locateAoiResult.loading}
      />
      <MenuItem
        intent={Intent.NONE}
        icon={IconNames.EXPORT}
        text={<Trans id="Export AOI geometry" />}
        onClick={exportAoiGeometry}
        disabled={getAoiResult.loading}
      />
    </Menu>
  );
};

ActionsMenu = React.memo(ActionsMenu);

export { ActionsMenu };
