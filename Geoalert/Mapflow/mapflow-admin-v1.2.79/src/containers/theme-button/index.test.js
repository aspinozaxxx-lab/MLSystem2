import React from "react";
import { InMemoryCache } from "apollo-cache-inmemory";
import { Classes } from "@blueprintjs/core";

import { fireEvent, waitForElement } from "test-utils";
import { render, cleanup } from "test-utils";
import { GET_THEME } from "hooks/use-theme";
import themeResolvers from "graphql/resolvers/theme";

import { ThemeNames } from "constants/common";
import ThemeButton from ".";

describe("ThemeButton", () => {
  let component;
  beforeEach(() => {
    component = <ThemeButton />;
  });
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot light", () => {
    const { asFragment } = render(component, {
      withTheme: { initialTheme: ThemeNames.LIGHT },
    });
    expect(asFragment()).toMatchSnapshot();
  });

  it("matches snapshot dark", () => {
    const { asFragment } = render(component, {
      withTheme: { initialTheme: ThemeNames.DARK },
    });
    expect(asFragment()).toMatchSnapshot();
  });

  it("do_gql_queries", async () => {
    const cache = new InMemoryCache();
    cache.writeData({ data: { theme: ThemeNames.LIGHT } });
    const { getByTestId, getByText, container } = render(component, {
      withApollo: { cache, resolvers: themeResolvers.resolvers },
    });

    const getThemed = () => container.querySelector(`.${Classes.DARK}`);
    expect(getThemed()).toBeNull();

    fireEvent.click(getByTestId("theme-button"));
    let toast = await waitForElement(() =>
      getByText(/You activated a dark theme/i),
    );
    expect(toast).toBeInTheDocument();
    expect(getThemed()).toBeInTheDocument();
    let response = cache.readQuery({ query: GET_THEME });
    expect(response.theme).toBe(ThemeNames.DARK);

    fireEvent.click(getByTestId("theme-button"));
    toast = await waitForElement(() =>
      getByText(/You activated a light theme/i),
    );
    expect(toast).toBeInTheDocument();
    expect(getThemed()).toBeNull();
    response = cache.readQuery({ query: GET_THEME });
    expect(response.theme).toBe(ThemeNames.LIGHT);
  });
});
