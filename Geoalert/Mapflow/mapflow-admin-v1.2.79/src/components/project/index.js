import React from "react";
import { Redirect, Route, Switch } from "react-router-dom";

import * as routes from "constants/routes";
import { Processings, Processing, LinkWorkflow, ProjectWorkflows } from "pages";
import { PrivateRoute } from "shared/routing/private-route";

function Project() {
  return (
    <Switch>
      <PrivateRoute
        exact
        path={routes.PROJECT_WORKFLOW_LINK}
        component={LinkWorkflow}
      />
      <Route exact path={routes.PROJECT_PROCESSING} component={Processing} />
      <Route
        exact
        path={routes.PROJECT_WORKFLOWS}
        component={ProjectWorkflows}
      />
      <Route exact path={routes.PROJECT_PROCESSINGS} component={Processings} />
      <Redirect to={routes.PROJECT_WORKFLOWS} />
    </Switch>
  );
}

export default React.memo(Project);
