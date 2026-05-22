import React from "react";

import { render, cleanup, wait } from "test-utils";
import InfiniteList from ".";

describe("InfiniteList", () => {
  let component;
  beforeEach(() => (component = <InfiniteList items={[]} />));
  afterEach(cleanup);

  it("renders without error", async () => {
    render(component);
    await wait();
  });
});
