import React from "react";
import { t, Trans } from "@lingui/macro";
import { Button, Intent, H5 } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import ConfirmDialog from "components/confirm-dialog";
import { gql, useApolloClient } from "@apollo/client";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { useParams } from "react-router-dom";

import CreateProcessingDialog from "components/create-processing-dialog";
import { showToast, getSuccessToast, getErrorToast } from "toaster";
import { setTestId } from "test-utils/set-testid";
import { ErrorCodes } from "constants/common";
import useProjectQuery from "./queries";

export const UNLINK_WORKFLOW_FROM_PROJECT = gql`
  mutation unlinkWorkflowDefFromProject($workflowDefId: ID!, $projectId: ID!) {
    unlinkWorkflowDefFromProject(
      workflowDefId: $workflowDefId
      projectId: $projectId
    )
  }
`;

function Actions({ workflowDefId, name, description, isWorkflowDefault, blocks }) {
  const { projectId } = useParams();

  const client = useApolloClient();
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationKey: ["unlinkWorkflowFromProject", workflowDefId],
    mutationFn: async (variables) => {
      const result = await client.mutate({
        mutation: UNLINK_WORKFLOW_FROM_PROJECT,
        fetchPolicy: "no-cache",
        variables: variables,
      });
      return result?.data?.unlinkWorkflowDefFromProject;
    },
    onSuccess: () => {
      queryClient.invalidateQueries(["projectWorkflows", projectId], {
        force: true,
      });

      showToast(
        getSuccessToast(t`Workflow unlinked`, {
          icon: IconNames.UNDO,
        }),
      );
    },
    onError: (error) => {
      const { graphQLErrors } = error;
      if (graphQLErrors)
        if (graphQLErrors.some(({ code }) => code === ErrorCodes.WD_IN_USE)) {
          showToast(
            getErrorToast(t`Workflow cannot be unlinked while it is in use`),
          );
          return;
        }
      showToast(getErrorToast(t`Error unlink workflow`));
    },
  });

  const { data: projectData } = useProjectQuery(projectId);
  const isProjectDefault = projectData?.isDefault;

  const unlinkAction = !isWorkflowDefault && !isProjectDefault && (
    <ConfirmDialog
      className="unlink-workflow"
      intent={Intent.DANGER}
      icon={IconNames.UNDO}
      confirmButtonText={<Trans id="Unlink" />}
      cancelButtonText={<Trans id="Cancel" />}
      text={
        <H5>
          <Trans id="Confirm unlink workflow" />
        </H5>
      }
      onConfirm={(close) => {
        mutation.mutate({
          workflowDefId,
          projectId,
        });
        close();
      }}
    >
      {({ showDialog }) => (
        <Button
          minimal
          elementRef={setTestId`unlink-workflow`}
          icon={IconNames.UNDO}
          intent={Intent.DANGER}
          text={<Trans id="Unlink" />}
          disabled={mutation.isLoading}
          onClick={showDialog}
        />
      )}
    </ConfirmDialog>
  );

  return (
    <>
      <div>
        <CreateProcessingDialog
          workflowDefId={workflowDefId}
          workflowName={name}
          workflowDesc={description}
          blocks={blocks}
        >
          {({ showDialog }) => (
            <Button
              minimal
              elementRef={setTestId`create-processing`}
              icon={IconNames.NEW_OBJECT}
              intent={Intent.PRIMARY}
              text={<Trans>Create processing</Trans>}
              onClick={showDialog}
            />
          )}
        </CreateProcessingDialog>

        {unlinkAction}
      </div>
    </>
  );
}

export default React.memo(Actions);
