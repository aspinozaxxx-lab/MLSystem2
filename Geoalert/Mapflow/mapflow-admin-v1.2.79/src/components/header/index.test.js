/* eslint-disable react/jsx-pascal-case */
import React from "react";

import { render, cleanup } from "test-utils";
import Header from ".";

describe("Header", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("renders without error", () => {
    const component = <Header />;
    render(component);
  });

  it("matches snapshot", () => {
    const component = <Header />;
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
