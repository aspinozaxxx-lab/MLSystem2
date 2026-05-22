/* eslint-disable react/jsx-pascal-case */
import React from "react";

import { render, cleanup, wait } from "test-utils";
import SignOutButton from ".";

describe("SignOutButton", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("renders without error", async () => {
    const component = <SignOutButton />;
    render(component);
    await wait();
  });

  it("matches snapshot", async () => {
    const component = <SignOutButton />;
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });
});
