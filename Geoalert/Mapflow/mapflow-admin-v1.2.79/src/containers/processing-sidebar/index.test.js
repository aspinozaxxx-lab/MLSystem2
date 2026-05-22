import React from "react";
import { QueryCache } from "@tanstack/react-query";

import { render, cleanup, within, waitForElement, wait } from "test-utils";
import { mockOffsetSize } from "test-utils/helpers";
import processing from "fixtures/processing";

import * as routes from "constants/routes";
import ProcessingSidebar from ".";

import { GET_PROJECT_NAME } from "containers/breadcrumbs";
import { GET_PROCESSING } from "containers/processing-sidebar";
import projects from "fixtures/projects";
const [project] = projects;

jest.mock("graphql/cache-redirects");
jest.mock("processing-list-mock");
jest.mock("project-list-mock");

describe("ProcessingSidebar", () => {
  let component;

  beforeEach(() => {
    mockOffsetSize(200, 300);
    component = <ProcessingSidebar />;
  });
  afterEach(cleanup);

  it("should show load sidebar correctly", async () => {
    const projectName = project["name"];
    const processingName = processing["name"];
    const processingDesc = processing["description"];
    const workflowName = processing["workflowDef"]["name"];
    const workflowDesc = processing["workflowDef"]["description"];
    const processingId = processing["id"];
    const projectId = "1";
    const path = routes.PROCESSING;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;
    const mocks = [
      {
        request: { query: GET_PROCESSING, variables: { processingId } },
        result: { data: { processing } },
      },
      {
        request: { query: GET_PROJECT_NAME, variables: { id: projectId } },
        result: { data: { project: { name: projectName } } },
      },
    ];
    const options = {
      withApollo: { mocks },
      withRouter: { route: processingRoute, path },
    };

    QueryCache.setQueryData(["processing", processingId], { ...processing });

    const { asFragment, getByText } = render(component, options);

    await wait();
    expect(asFragment()).toMatchSnapshot();
    // check processing name and workflow info
    await waitForElement(() => getByText(processingName));
    await waitForElement(() => getByText(processingDesc));
    await waitForElement(() => getByText(workflowName));
    await waitForElement(() => getByText(workflowDesc));
  });
});
