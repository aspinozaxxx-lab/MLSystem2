import React from "react";

import { render, cleanup, wait } from "test-utils";
import MapLoader from ".";

describe("MapLoader", () => {
  let component;
  beforeEach(() => (component = <MapLoader isDataLoading />));
  afterEach(cleanup);

  it("renders without error", async () => {
    render(component);
    await wait();
  });

  it("matches snapshot dataloading", async () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });

  it("matches snapshot maploading", async () => {
    const { asFragment } = render(<MapLoader isMapLoading />);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });
});
