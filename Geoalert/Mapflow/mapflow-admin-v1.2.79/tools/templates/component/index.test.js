import React from "react";

import { render, cleanup, wait } from "test-utils";
import __name__ from ".";

describe("__name__", () => {
  let component;
  beforeEach(() => (component = <__name__ />));
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
