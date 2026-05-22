import React, { useMemo, useState } from "react";
import { gql } from "@apollo/client";
import { Trans, t } from "@lingui/macro";
import { IconNames } from "@blueprintjs/icons";
import {
  Intent,
  Position,
  Button,
  ButtonGroup,
  Popover,
} from "@blueprintjs/core";
import { useParams } from "react-router-dom";

import { useTheme } from "hooks/use-theme";

import { useMutationWithToasts } from "../../hooks/use-mutation-with-toasts/index";
import { ActionsMenu } from "./actions-menu";
import { setTestId } from "test-utils/set-testid";

export const RUN_PROCESSING = gql`
  mutation runProcessing($id: ID!) {
    runProcessing(processingId: $id)
  }
`;

function AoiListActions({
  onUpload,
  progress,
  isGeotiffDataProvider,
  isUnproceesedAois,
  aoiCount,
  isFailed,
  isPending,
}) {
  const { processingId } = useParams();

  const [processingCommand, setProcessingCommand] = useState(false);

  const updateProcessingCommand = () => {
    setTimeout(() => {
      setProcessingCommand(false);
    }, 2000);
  };

  const variables = useMemo(
    () => ({
      filter: { processingIds: [processingId] },
    }),
    [processingId],
  );

  const [runProcessing, runProcessingResult] = useMutationWithToasts(
    RUN_PROCESSING,
    {
      options: { variables: { id: processingId }, refetchQueries: ["getAois"] },
      getSuccesMessage: ({ data }) => t`Successfully started processing`,
      getDefaultErrorMessage: () => t`Error run processing`,
      pendingIntent: Intent.PRIMARY,
      pendingIcon: IconNames.PREDICTIVE_ANALYSIS,
    },
    updateProcessingCommand,
  );

  const actionsMenu = useMemo(
    () => (
      <ActionsMenu
        isGeotiffDataProvider={isGeotiffDataProvider}
        onUpload={onUpload}
        filter={variables.filter}
        isFailed={isFailed}
        isPending={isPending}
        processingCommand={processingCommand}
        updateProcessingCommand={updateProcessingCommand}
        setProcessingCommand={setProcessingCommand}
      />
    ),
    [
      isFailed,
      isGeotiffDataProvider,
      isPending,
      onUpload,
      processingCommand,
      variables.filter,
    ],
  );

  const { themeClassName } = useTheme();

  return (
    <div className="aoi-list-actions">
      <ButtonGroup>
        <Button
          fill
          large
          elementRef={setTestId`run-processing-all`}
          className="run-processing-button"
          icon={IconNames.PREDICTIVE_ANALYSIS}
          intent={Intent.PRIMARY}
          text={<Trans id="Run processing" />}
          onClick={() => {
            setProcessingCommand(true);
            runProcessing();
          }}
          disabled={
            runProcessingResult.loading ||
            !isUnproceesedAois ||
            processingCommand
          }
        />
        <Popover
          className={themeClassName}
          position={Position.TOP_LEFT}
          content={actionsMenu}
        >
          <Button
            large
            elementRef={setTestId`show-processing-actions`}
            intent={Intent.PRIMARY}
            icon={IconNames.MORE}
            disabled={runProcessingResult.loading}
          />
        </Popover>
      </ButtonGroup>
    </div>
  );
}

export default React.memo(AoiListActions);
