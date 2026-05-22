import React from "react";
import { cleanup } from "@testing-library/react";

import { render } from "test-utils";
import Loading from ".";

describe("Loading", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("renders without error", () => {
    const component = <Loading />;
    render(component);
  });

  it("matches snapshot", () => {
    const component = <Loading />;
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
