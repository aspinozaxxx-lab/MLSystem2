import React from "react";

import { render, cleanup } from "test-utils";
import EmptyMessage from ".";

describe("EmptyMessage", () => {
  let component;
  beforeEach(() => (component = <EmptyMessage />));
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
