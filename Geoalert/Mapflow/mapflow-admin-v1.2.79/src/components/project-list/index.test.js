import React from "react";

import { render, cleanup } from "test-utils";
import ProjectList from ".";

import projectList from "project-list-mock";
jest.mock("project-list-mock");

describe("ProjectList", () => {
  let component;
  beforeEach(() => (component = <ProjectList projects={projectList} />));
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });
});
