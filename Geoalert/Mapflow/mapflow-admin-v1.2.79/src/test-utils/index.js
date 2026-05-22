import React from "react";
import { render } from "@testing-library/react";
import { ReactQueryConfigProvider } from "@tanstack/react-query";
import "@testing-library/jest-dom/extend-expect";

import { withRouter, withApollo, withTheme, withI18n } from "./hocs";

import compose from "ramda/src/compose";
import mapObjIndexed from "ramda/src/mapObjIndexed";
import call from "ramda/src/call";

const queryConfig = { queries: { retry: false } };

const Providers = {
  withApollo,
  withI18n,
  withRouter,
  withTheme,
};

const getUserProps = (options) => {
  const { withApollo, withRouter, withTheme, withI18n, ...rest } = options;
  return { props: { withApollo, withRouter, withTheme, withI18n }, rest };
};

const getProviderList = (options) => {
  const { getProviders = (d) => d } = options;
  const { props, rest } = getUserProps(options);
  const defaultProviders = mapObjIndexed((hocFn, name) => {
    if (name in props) return call(hocFn, props[name]);
    else return call(hocFn);
  }, Providers);
  const updatedProviders = getProviders(defaultProviders);
  const providersList = Object.values(updatedProviders);
  return { providersList, restProps: rest };
};

const createWrapper = (providers) => {
  return compose(...providers)(({ children }) => (
    <ReactQueryConfigProvider config={queryConfig}>
      {children}
    </ReactQueryConfigProvider>
  ));
};

const customRender = (ui, options = {}) => {
  const { providersList, restProps } = getProviderList(options);
  const wrapper = createWrapper(providersList);
  return render(ui, { wrapper, ...restProps });
};

// re-export everything
export * from "@testing-library/react";

// override render method
export { customRender as render, Providers };
