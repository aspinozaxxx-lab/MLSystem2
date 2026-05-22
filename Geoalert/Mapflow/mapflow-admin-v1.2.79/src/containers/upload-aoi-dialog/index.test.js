import React from "react";
import { render, cleanup, wait } from "test-utils";

import UploadAoiDialog from ".";

// TODO add tests
describe("UploadAoiDialog", () => {
  let component;
  beforeEach(() => {
    component = <UploadAoiDialog />;
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
