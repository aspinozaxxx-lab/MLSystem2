import React from "react";

import { Redirect, Route, Switch } from "react-router-dom";

import * as routes from "constants/routes";

import DataProvider from "pages/data-provider";
import CreateEditDataProvider from "pages/create-edit-data-provider";

import ManageDataProviderUsers from 'pages/manage-data-provider-users'



function DataProvidersRoutes() {
  return (
    <Switch>
      <Route
        exact
        path={routes.DATA_PROVIDER_CREATE}
        component={CreateEditDataProvider}
      />
      <Route exact path={routes.DATA_PROVIDER_EDIT} component={CreateEditDataProvider} />
   
      <Route
        exact
        path={routes.DATA_PROVIDER_USERS}
        component={ManageDataProviderUsers}
      />
      <Route exact path={routes.DATA_PROVIDERS} component={DataProvider} />

      <Redirect from={routes.DATA_PROVIDERS} to={routes.DATA_PROVIDERS} />
    </Switch>
  );
}

export default  React.memo(DataProvidersRoutes);
