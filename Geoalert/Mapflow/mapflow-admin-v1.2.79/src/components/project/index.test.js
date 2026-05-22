import React from "react";

import { render, cleanup } from "test-utils";
import Project from ".";

jest.mock("../../containers/breadcrumbs");

describe("Project", () => {
  let component;
  beforeEach(() => {
    component = <Project />;
    Element.prototype.getBoundingClientRect = jest.fn(() => {
      return { width: 520, height: 220 };
    });
  });
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
