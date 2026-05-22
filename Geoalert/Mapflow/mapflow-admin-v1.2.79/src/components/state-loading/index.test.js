import React from "react";
import { cleanup } from "@testing-library/react";

import { render } from "test-utils";
import StateLoading from ".";

describe("StateLoading", () => {
  let component;
  beforeEach(() => {
    component = <StateLoading title="title" description="description" />;
  });
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
