import React from "react";
import { render, cleanup, wait } from "test-utils";

import CreateWorkflowFromFile from ".";

// TODO add tests
describe("CreateWorkflowFromFile", () => {
  let component;
  beforeEach(() => {
    component = <CreateWorkflowFromFile />;
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
