import React from "react";

import { render, cleanup, wait } from "test-utils";
import App from ".";

describe("App", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("renders without error", async () => {
    const component = <App />;
    render(component);
    await wait();
  });

  // it("matches snapshot", async () => {
  //   const component = <App />;
  //   const { asFragment } = render(component);
  //   expect(asFragment()).toMatchSnapshot();
  //   await wait();
  // });
});
