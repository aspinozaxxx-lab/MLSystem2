import React from "react";

import { render, cleanup, wait } from "test-utils";
import StatusTag from ".";

describe("StatusTag", () => {
  let component;
  beforeEach(() => (component = <StatusTag />));
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
