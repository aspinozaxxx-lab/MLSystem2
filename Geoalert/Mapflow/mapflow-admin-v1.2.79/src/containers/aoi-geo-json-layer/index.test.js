import React from "react";
import { render, cleanup, wait } from "test-utils";

import AoiGeoJsonLayer from ".";

// TODO add tests
describe("AoiGeoJsonLayer", () => {
  let component;
  beforeEach(() => {
    component = <AoiGeoJsonLayer />;
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
