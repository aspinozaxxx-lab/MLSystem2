import React from "react";
import { Redirect, Router } from "react-router";
import { Route, Switch } from "react-router-dom";

import * as routes from "constants/routes";
import { Header, Workflows, DataProviderRoutes, Project } from "components";
import { history } from "router-history";
import { ProcessingsStats, Projects } from "pages";
import { MainNavbar } from "containers";
import { PrivateRoute } from "shared/routing/private-route";

const WorkflowSearchContext = React.createContext({
  search: "",
  setSearch: () => {},
});
export const useWorkflowSearch = () => React.useContext(WorkflowSearchContext);

const ProjectSearchContext = React.createContext({
  search: "",
  setSearch: () => {},
});
export const useProjectSearch = () => React.useContext(ProjectSearchContext);

function App() {
  const [workflowSearch, setWorkflowSearch] = React.useState("");
  const [projectSearch, setProjectSearch] = React.useState("");

  return (
    <ProjectSearchContext.Provider
      value={{ search: projectSearch, setSearch: setProjectSearch }}
    >
      <WorkflowSearchContext.Provider
        value={{ search: workflowSearch, setSearch: setWorkflowSearch }}
      >
        <Router history={history}>
          <Header />
          <div className="app-scroll">
            <MainNavbar />
            <Switch>
              <Route path={routes.PROJECT} component={Project} />
              <Route exact path={routes.PROJECTS} component={Projects} />

              <PrivateRoute path={routes.WORKFLOWS} component={Workflows} />

              <PrivateRoute
                path={routes.DATA_PROVIDERS}
                component={DataProviderRoutes}
              />

              <PrivateRoute
                path={routes.PROCESSING_STATS}
                component={ProcessingsStats}
              />

              <Redirect exact from={routes.MAIN} to={routes.PROJECTS} />
            </Switch>
          </div>
        </Router>
      </WorkflowSearchContext.Provider>
    </ProjectSearchContext.Provider>
  );
}

export default React.memo(App);
