import React from "react";

import { Redirect, Route, Switch } from "react-router-dom";

import * as routes from "constants/routes";

import { CreateEditWorkflow, ManageWorkflowUsers, WorkflowList } from "pages";

function Workflows() {
  return (
    <Switch>
      <Route
        exact
        path={routes.WORKFLOW_CREATE}
        component={CreateEditWorkflow}
      />
      <Route exact path={routes.WORKFLOW_EDIT} component={CreateEditWorkflow} />
      <Route
        exact
        path={routes.WORKFLOW_USERS}
        component={ManageWorkflowUsers}
      />
      <Route exact path={routes.WORKFLOWS} component={WorkflowList} />

      <Redirect from={routes.WORKFLOWS} to={routes.WORKFLOWS} />
    </Switch>
  );
}

export default React.memo(Workflows);
