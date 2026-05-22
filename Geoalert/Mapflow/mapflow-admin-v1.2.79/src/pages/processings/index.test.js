import React from "react";

import { render, cleanup, wait } from "test-utils";

import Processings from ".";

describe("Processings", () => {
  let component;
  beforeEach(() => {
    component = <Processings />;
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
