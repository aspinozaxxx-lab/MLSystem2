import React from "react";
import { render, cleanup, wait } from "test-utils";

import AoiListHeader from ".";

describe("AoiListHeader", () => {
  let component;
  beforeEach(() => {
    component = <AoiListHeader />;
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
