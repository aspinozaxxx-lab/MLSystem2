import React, { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";
import { useParams } from "react-router-dom";
import pathOr from "ramda/src/pathOr";
import {
  H2,
  Callout,
  Intent,
  Icon,
  Colors,
  Collapse,
  Button,
} from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

import { AoiList, AoiListActions } from "containers";
import { TextLoader, Subtitle, EditableNameInput } from "components";
import { POLL_INTERVAL } from "constants/envs";
import { Trans } from "@lingui/react";
import { UPDATE_PROCESSING_NAME } from "shared/api/project";
import { t } from "@lingui/macro";
import { Statuses } from "constants/common";

export const GET_PROCESSING = gql`
  query getProcessing($processingId: ID!) {
    processing(id: $processingId) {
      name
      description
      blocks {
        name
        displayName
        enabled
      }
      dataProvider {
        name
        displayName
        previewUrl
      }
      aoiCount
      bbox
      sourceType
      rasterLayer {
        id
        tileUrl
      }
      vectorLayer {
        tileJsonUrl
      }
      progress {
        status
        percentCompleted
        details {
          status
          area
        }
      }
      workflowDef {
        name
        description
      }
      meta
    }
  }
`;

function ProcessingSidebar({ openUploadAoiDialog, aoiResult }) {
  const { processingId } = useParams();

  const [showed, setShowed] = useState(true);

  const toggleShow = () => setShowed((prev) => !prev);

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
    refetchInterval: POLL_INTERVAL,
  });

  const loading = status === "loading";

  const processingName = pathOr(null, ["name"])(data);
  const processingDesc = pathOr(null, ["description"])(data);
  const processingProgress = pathOr(null, ["progress"])(data);
  const aoiCount = pathOr(null, ["aoiCount"])(data);
  const workflowName = pathOr(null, ["workflowDef", "name"])(data);
  const workflowDesc = pathOr(null, ["workflowDef", "description"])(data);
  const blocks = pathOr(null, ["blocks"])(data);
  const dataProvider = pathOr(null, ["dataProvider"])(data);
  const isGeotiffDataProvider =
    pathOr(null, ["dataProvider", "name"])(data) === "GTIFF";
  const filteredBlocks = blocks?.filter((block) => block.enabled);
  const meta = JSON.parse(pathOr(null, ["meta"])(data));

  const isUnproceesedAois = useMemo(() => {
    return aoiResult?.data?.some(
      (aoi) => aoi.progress.status === Statuses.UNPROCESSED,
    );
  }, [aoiResult]);

  const isFailed = useMemo(() => {
    return aoiResult?.data?.some(
      (aoi) =>
        aoi.progress.status === Statuses.FAILED ||
        aoi.progress.status === Statuses.STATUS_CANCELLED,
    );
  }, [aoiResult?.data]);

  const isPending = useMemo(() => {
    return aoiResult?.data?.some(
      (aoi) => aoi.progress.status === Statuses.PENDING,
    );
  }, [aoiResult?.data]);
  return (
    <div className="processing-sidebar">
      <div className="processing-sidebar__header">
        <div className="processing-sidebar__header-top">
          <div className="processing-title">
            <H2 className="processing-sidebar__name">
              <EditableNameInput
                value={processingName}
                mutRequest={UPDATE_PROCESSING_NAME}
                mutKey={["updateProcessingName", processingId]}
                mutationVariables={{ processingId }}
                successMessage={t`Processing name updated`}
                errorMessage={t`Error update processing name`}
                field={"name"}
                refetchQueryKey={["processing", "name", processingId]}
              />
            </H2>
            <Subtitle marginLeft="1" fontWeight="600">
              <TextLoader
                skip={!loading && !processingDesc}
                length="25"
                text={processingDesc}
              />
            </Subtitle>
          </div>
          <Button
            className="processing-sidebar__header-top__collapse-button"
            onClick={toggleShow}
            icon={!showed ? IconNames.ChevronDown : IconNames.ChevronUp}
            outlined
          />
        </div>

        <Collapse isOpen={showed}>
          <Callout
            intent={Intent.PRIMARY}
            icon={IconNames.APPLICATION}
            title={<TextLoader length="20" text={workflowName} />}
            style={{ marginTop: showed ? "20px" : "0" }}
          >
            <Subtitle marginLeft="1" fontWeight="600">
              <TextLoader
                ellipsize
                skip={!loading && !workflowDesc}
                length="15"
                text={workflowDesc}
              />
            </Subtitle>
          </Callout>

          {dataProvider && (
            <Callout
              intent={Intent.PRIMARY}
              icon={IconNames.GLOBE}
              className="processing-colout"
              title={<Trans id="Data Provider" />}
            >
              <Subtitle marginLeft="1" fontWeight="600">
                {dataProvider?.name}
              </Subtitle>

              {meta && meta?.rest.image_id && meta?.rest.image_date && (
                <>
                  <Subtitle marginLeft="1" marginTop="5" fontWeight="600">
                    Идентификатор изображения: {meta?.rest.image_id}
                  </Subtitle>
                  <Subtitle marginLeft="1" marginTop="5" fontWeight="600">
                    Дата съемки: {meta?.rest.image_date}
                  </Subtitle>
                </>
              )}
            </Callout>
          )}

          {filteredBlocks && filteredBlocks.length > 0 && (
            <Callout
              intent={Intent.PRIMARY}
              icon={IconNames.LIST}
              className="processing-colout"
              title={<Trans id="Options" />}
            >
              {filteredBlocks.map((block) => (
                <div className="processing-blocks__element" key={block.name}>
                  <Icon color={Colors.GREEN1} icon={"tick-circle"} />
                  <Subtitle marginLeft="1" marginRight="10" fontWeight="600">
                    {block.displayName || block.name}
                  </Subtitle>
                </div>
              ))}
            </Callout>
          )}
        </Collapse>
      </div>
      <div className="processing-sidebar__body">
        <AoiList
          isGeotiffDataProvider={isGeotiffDataProvider}
          aoiCount={aoiCount}
          onUpload={openUploadAoiDialog}
          aoiResult={aoiResult}
        />
        <AoiListActions
          progress={processingProgress}
          isUnproceesedAois={isUnproceesedAois}
          aoiCount={aoiCount}
          onUpload={openUploadAoiDialog}
          isGeotiffDataProvider={isGeotiffDataProvider}
          isFailed={isFailed}
          isPending={isPending}
        />
      </div>
    </div>
  );
}

export default React.memo(ProcessingSidebar);
