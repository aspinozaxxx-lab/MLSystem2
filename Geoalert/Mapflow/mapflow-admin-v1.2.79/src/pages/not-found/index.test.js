import React from "react";
import { render, cleanup, wait } from "test-utils";

import NotFound from ".";

describe("NotFound", () => {
  let component;
  beforeEach(() => {
    component = <NotFound />;
  });
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
