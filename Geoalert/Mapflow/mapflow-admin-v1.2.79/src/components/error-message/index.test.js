import React from "react";

import { render, cleanup } from "test-utils";
import ErrorMessage from ".";

describe("ErrorMessage", () => {
  let component;
  beforeEach(() => (component = <ErrorMessage />));
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
