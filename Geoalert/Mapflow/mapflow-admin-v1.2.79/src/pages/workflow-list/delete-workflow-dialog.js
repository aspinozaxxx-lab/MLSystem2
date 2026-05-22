import React from "react";
import { Trans, t } from "@lingui/macro";
import { Button, Intent, H5, Text, ProgressBar } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import ConfirmDialog from "components/confirm-dialog";
import { useGoTo } from "hooks/use-go-to";
import * as routes from "constants/routes";

import {
  useUsersLinkedToWorkflowQuery,
  useDeleteWorkflowMutation,
} from "./queries";

function WorkflowUsersCount({ workflowDefId, goToWorkflowUsers }) {
  const {
    data: users,
    isLoading: usersLoading,
  } = useUsersLinkedToWorkflowQuery(workflowDefId);

  return users?.length > 0 ? (
    <div onClick={goToWorkflowUsers} className="unlink-workflow__users">
      <Text className="unlink-workflow__linked">
        <Trans id={`Linked users: `} />
      </Text>

      {usersLoading ? (
        <ProgressBar intent={Intent.PRIMARY} />
      ) : (
        <div className="unlink-workflow__user-emails">
          {users.map((user) => (
            <Text key={user.email} className="">
              {user.email}
            </Text>
          ))}
        </div>
      )}
    </div>
  ) : (
    <Text>
      <Trans id={`No linked users`} />
    </Text>
  );
}

function DeleteWorkflowDialog({ workflowDefId, name, ...btnProps }) {
  const goToWorkflowUsers = useGoTo(routes.WORKFLOW_USERS, {
    workflowDefId,
  });

  const deleteWorkflowMutation = useDeleteWorkflowMutation(workflowDefId, name);

  return (
    <ConfirmDialog
      className="unlink-workflow"
      intent={Intent.DANGER}
      icon={IconNames.TRASH}
      confirmButtonText={<Trans id="Delete" />}
      cancelButtonText={<Trans id="Cancel" />}
      text={
        <div>
          <H5>
            <Trans>{t`Confirm delete workflow "${name}"`}</Trans>
          </H5>

          <WorkflowUsersCount
            workflowDefId={workflowDefId}
            goToWorkflowUsers={goToWorkflowUsers}
          />
        </div>
      }
      onConfirm={(close) => {
        deleteWorkflowMutation.mutate({
          workflowDefId,
        });
        close();
      }}
    >
      {({ showDialog }) => (
        <Button
          icon={IconNames.TRASH}
          intent={Intent.DANGER}
          text={<Trans id="Delete" />}
          disabled={deleteWorkflowMutation.isLoading}
          onClick={showDialog}
          {...btnProps}
        />
      )}
    </ConfirmDialog>
  );
}

export default React.memo(DeleteWorkflowDialog);
