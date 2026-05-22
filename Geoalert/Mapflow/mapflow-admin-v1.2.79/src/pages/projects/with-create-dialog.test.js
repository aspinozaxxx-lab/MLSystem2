import React from "react";
import { render, waitForElement, cleanup, wait, act } from "test-utils";
import projects from "fixtures/projects";

import { createMemoryHistory } from "history";

import { withCreateDialog } from "./with-create-dialog";
import {
  _isOpen,
  _isLoading,
  _createProject,
} from "components/create-project-dialog";

import client from "graphql/client";
import { responseWithDelay } from "test-utils/response-with-delay";
import { QueryCache } from "@tanstack/react-query";
jest.mock("graphql/client");

const [card] = projects;
jest.mock("graphql/client");
jest.mock("components/create-project-dialog");

describe("withCreateDialog", () => {
  let project, Component, _onCreateProject;

  beforeEach(() => {
    [project] = projects;
    Component = function ({ onCreateProject }) {
      _onCreateProject = onCreateProject;
      return null;
    };
  });

  afterEach(() => {
    jest.clearAllMocks();
    cleanup();
  });

  it("should create project", async () => {
    const history = createMemoryHistory({ initialEntries: ["/"] });
    QueryCache.setQueryData("projects", []);
    client.mutate.mockImplementationOnce(
      responseWithDelay({ data: { createProject: card } }),
    );
    const WithCreateDialog = withCreateDialog(Component);
    const { getByText } = render(<WithCreateDialog />, {
      withRouter: { history },
    });
    expect(_isOpen).toHaveBeenNthCalledWith(1, false);
    expect(_isLoading).toHaveBeenNthCalledWith(1, false);
    act(() => {
      _onCreateProject();
    });
    expect(_isOpen).toHaveBeenNthCalledWith(2, true);

    act(() => {
      _createProject({ name: project.name });
    });
    expect(_isLoading).toHaveBeenNthCalledWith(2, true);
    await wait();
    await waitForElement(() =>
      getByText(`Project "${project.name}" successfully created`),
    );
    expect(history.location.pathname).toBe(`/projects/${project.id}/workflows`);
    await wait();
  });

  it("should create project with error", async () => {
    client.query.mockImplementationOnce(
      responseWithDelay(null, { error: "Error" }),
    );
    const WithCreateDialog = withCreateDialog(Component);
    const { getByText } = render(<WithCreateDialog />);
    act(() => {
      _createProject({ name: project.name });
    });
    await waitForElement(() => getByText(`Error creating project`));
  });
});
