import React from "react";

import { render, cleanup, wait } from "test-utils";
import SortIndicator from ".";

describe("SortIndicator", () => {
  let component;
  beforeEach(() => (component = <SortIndicator />));
  afterEach(cleanup);

  it("renders without error", async () => {
    render(component);
    await wait();
  });

  it("matches snapshot", async () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });
});
