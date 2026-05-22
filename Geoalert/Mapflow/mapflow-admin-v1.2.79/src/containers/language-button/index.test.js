import React from "react";
import { InMemoryCache } from "apollo-cache-inmemory";

import { cleanup, render, fireEvent, waitForElement } from "test-utils";
import languageResolvers from "graphql/resolvers/language";
import { GET_LANGUAGE } from "hooks/use-language";
import LanguageButton from ".";
import { LanguageNames } from "constants/common";

describe("LanguageButton", () => {
  let component;
  beforeEach(() => (component = <LanguageButton />));
  afterEach(cleanup);

  it("renders without error", () => {
    const cache = new InMemoryCache();
    cache.writeData({ data: { language: LanguageNames.EN } });
    render(component, { withApollo: { cache } });
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });

  it("should toggle language", async () => {
    const cache = new InMemoryCache();
    cache.writeData({ data: { language: LanguageNames.EN } });
    const { getByTestId, getByText } = render(component, {
      withApollo: { cache, resolvers: languageResolvers.resolvers },
    });

    fireEvent.click(getByTestId("language-button"));
    await waitForElement(() => getByText(/ru/i));
    let response = cache.readQuery({ query: GET_LANGUAGE });
    expect(response.language).toBe("ru");

    fireEvent.click(getByTestId("language-button"));
    await waitForElement(() => getByText(/en/i));
    response = cache.readQuery({ query: GET_LANGUAGE });
    expect(response.language).toBe(LanguageNames.EN);
  });
});
