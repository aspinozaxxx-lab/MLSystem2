import React from "react";
import { Trans } from "@lingui/macro";
import { Button, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

import { setTestId } from "test-utils/set-testid";
import * as routes from "constants/routes";
import { useGoTo } from "hooks/use-go-to";
import DeleteWorkflowDialog from "./delete-workflow-dialog";
import { Link, generatePath } from "react-router-dom";

function Actions({ workflowDefId, isDefault , name }) {
  const goToWorkflowUsers = useGoTo(routes.WORKFLOW_USERS, {
    workflowDefId,
  });
  const goToWorkflowEdit = useGoTo(routes.WORKFLOW_EDIT, {
    workflowDefId,
  });

  return (
    <div>
      <Link to={generatePath(routes.WORKFLOW_EDIT, { workflowDefId })}>
        <Button
          minimal
          elementRef={setTestId`edit-workflow`}
          icon={IconNames.EDIT}
          intent={Intent.SUCCESS}
          text={<Trans id="Edit" />}
          onClick={goToWorkflowEdit}
        />
      </Link>

      {!isDefault && (
        <Link to={generatePath(routes.WORKFLOW_USERS, { workflowDefId })}>
          <Button
            minimal
            elementRef={setTestId`manage-users`}
            icon={IconNames.USER}
            intent={Intent.PRIMARY}
            text={<Trans id="Manage users" />}
            onClick={goToWorkflowUsers}
          />
        </Link>
      )}

      <DeleteWorkflowDialog minimal workflowDefId={workflowDefId} name ={name} />
    </div>
  );
}

export default React.memo(Actions);
