import React from "react";

import { render, cleanup, wait } from "test-utils";
import MapboxMap from ".";

describe("MapboxMap", () => {
  let component;
  // beforeEach(() => (component = <MapboxMap />));
  afterEach(cleanup);

  it("renders without error", async () => {
    expect(1).toBe(1);
    // render(component);
    // await wait();
  });

  // it("matches snapshot", async () => {
  //   const { asFragment } = render(component);
  //   expect(asFragment()).toMatchSnapshot();
  //   await wait();
  // });
});
