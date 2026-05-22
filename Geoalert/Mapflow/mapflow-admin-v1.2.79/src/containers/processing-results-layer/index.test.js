import React from "react";
import { render, cleanup, wait } from "test-utils";

import ProcessingResultsLayer from ".";

import mapboxGl from "mapbox-gl";

// TODO add tests
describe("ProcessingResultsLayer", () => {
  let component;
  beforeEach(() => {
    component = (
      <ProcessingResultsLayer
        mapAPI={mapboxGl.Map({ zoom: 0, center: [0, 0] })}
      />
    );
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
