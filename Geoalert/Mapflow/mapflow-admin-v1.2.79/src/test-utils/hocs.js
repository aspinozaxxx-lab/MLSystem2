import React from "react";
import { Router, Route } from "react-router-dom";
import { createMemoryHistory } from "history";
import { MockedProvider } from "@apollo/react-testing";

import { ThemeProvider } from "providers/ThemeProvider";
import { I18nLoader } from "providers/LanguageProvider";

export const withRouter = ({
  route = "/",
  path = "/",
  history = createMemoryHistory({ initialEntries: [route] }),
} = {}) => (WrappedComponent) => (props) => {
  return (
    <>
      <Router history={history}>
        <Route path={path}>
          <WrappedComponent {...props} />
        </Route>
      </Router>
    </>
  );
};

export const withApollo = ({
  mocks = [],
  resolvers = {},
  addTypename = false,
  defaultOptions,
  cache,
} = {}) => (WrappedComponent) => (props) => (
  <>
    <MockedProvider
      mocks={mocks}
      addTypename={addTypename}
      defaultOptions={defaultOptions}
      cache={cache}
      resolvers={resolvers}
    >
      <WrappedComponent {...props} />
    </MockedProvider>
  </>
);

export const withTheme = ({ initialTheme } = {}) => (WrappedComponent) => (
  props,
) => (
  <>
    <ThemeProvider initialTheme={initialTheme}>
      <WrappedComponent {...props} />
    </ThemeProvider>
  </>
);

export const withI18n = ({ initialLanguage } = {}) => (WrappedComponent) => (
  props,
) => {
  return (
    <>
      <I18nLoader initialLanguage={initialLanguage}>
        <WrappedComponent {...props} />
      </I18nLoader>
    </>
  );
};
