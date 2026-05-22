import React from "react";

import { render, cleanup, wait, waitForElement } from "test-utils";

import Breadcrumbs, { GET_PROJECT_NAME, GET_PROCESSING_NAME } from ".";
import * as routes from "constants/routes";
//import { GET_PROJECTS } from "pages/projects/queries";
import { GET_PROCESSINGS } from "components/processing-list";

import projectList from "fixtures/projects";
import processingList from "fixtures/processings";
import { QueryCache } from "@tanstack/react-query";

// TODO add test - breadcrumbs should not crashes when no match or all params is nullable
describe("Breadcrumbs", () => {
  let component;
  let projectName = "project-1",
    processingName = "processing-1",
    processingId = 1,
    projectId = 1,
    cache;

  beforeAll(() => {
    QueryCache.setQueryData(["project", "name", projectId], {
      project: { name: projectName },
    });
    QueryCache.setQueryData(["processing", "name", processingId], {
      project: { name: processingName },
    });
  });

  beforeEach(() => {
    component = <Breadcrumbs />;
    Element.prototype.getBoundingClientRect = jest.fn(() => {
      return { width: 520, height: 220 };
    });
  });
  afterEach(cleanup);

  it("renders without error", async () => {
    const route = `/projects/${projectId}`;
    const path = routes.PROJECT;

    const options = { withRouter: { route, path } };
    render(component, options);
    await wait();
  });

  // it("should show project breadcrumbs", async () => {
  //   const route = `/projects/${projectId}`;
  //   const path = routes.PROJECT;
  //   const options = {
  //     withRouter: { route, path },
  //   };
  //   const { getByText, asFragment } = render(component, options);
  //   await waitForElement(() => getByText("Projects"));
  //   await waitForElement(() => getByText(projectName));
  //   expect(asFragment()).toMatchSnapshot();
  // });

  // // TODO add test where processing name gets from GET_PROCESSING query
  // it("should show processing breadcrumbs from processing-list cache", async () => {
  //   const route = `/projects/${projectId}/processings/${processingId}`;
  //   const path = routes.PROCESSING;

  //   const options = {
  //     withRouter: { route, path },
  //   };
  //   const { getByText, asFragment } = render(component, options);
  //   await waitForElement(() => getByText("Projects"));
  //   await waitForElement(() => getByText(projectName));
  //   await waitForElement(() => getByText("Processings"));
  //   await waitForElement(() => getByText(processingName));
  //   expect(asFragment()).toMatchSnapshot();
  // });

  // it("should get from cache first", async () => {
  //   const route = `/projects/${projectId}`;
  //   const path = routes.PROJECT;
  //   const getProjectMockFn = jest.fn(() => ({
  //     data: { project: { name: "NameFromServer" } },
  //   }));
  //   const mocks = [
  //     {
  //       request: { query: GET_PROJECT_NAME, variables: { id: projectId } },
  //       result: getProjectMockFn,
  //     },
  //   ];
  //   cache.writeQuery({
  //     query: GET_PROJECTS,
  //     data: { projects: projectList, __typename: "Projects" },
  //   });
  //   const options = {
  //     withRouter: { route, path },
  //     withApollo: { mocks, cache },
  //   };
  //   const { getByText } = render(component, options);
  //   await waitForElement(() => getByText("Projects"));
  //   await waitForElement(() => getByText(projectName));
  // });

  // it("should get from server if no data in the cache", async () => {
  //   const route = `/projects/${projectId}`;
  //   const path = routes.PROJECT;
  //   const getProjectMockFn = jest.fn(() => ({
  //     data: { project: { name: "NameFromServer" } },
  //   }));
  //   const mocks = [
  //     {
  //       request: { query: GET_PROJECT_NAME, variables: { id: projectId } },
  //       result: getProjectMockFn,
  //     },
  //   ];
  //   const options = {
  //     withRouter: { route, path },
  //     withApollo: { mocks },
  //   };
  //   const { getByText } = render(component, options);
  //   await waitForElement(() => getByText("Projects"));
  //   await waitForElement(() => getByText("NameFromServer"));
  // });

  // it("should show minus if do not get name from anywhere", async () => {
  //   const route = `/projects/${projectId}/processings/${processingId}`;
  //   const path = routes.PROCESSING;
  //   const getProjectMockFn = jest.fn(() => ({
  //     data: { project: { name: null } },
  //   }));
  //   const getProcessingMockFn = jest.fn(() => ({
  //     data: { processing: { name: null } },
  //   }));
  //   const mocks = [
  //     {
  //       request: { query: GET_PROJECT_NAME, variables: { id: projectId } },
  //       result: getProjectMockFn,
  //     },
  //     {
  //       request: {
  //         query: GET_PROCESSING_NAME,
  //         variables: { id: processingId },
  //       },
  //       result: getProcessingMockFn,
  //     },
  //   ];
  //   const options = {
  //     withRouter: { route, path },
  //     withApollo: { mocks },
  //   };
  //   const { getByText, getAllByText } = render(component, options);
  //   await waitForElement(() => getByText("Projects"));
  //   await waitForElement(() => getByText("Processings"));
  //   const minuses = getAllByText("-");
  //   expect(minuses).toHaveLength(2);

  //   expect(getProjectMockFn).toHaveBeenCalled();
  //   // calls cache only because getProcessingName query has @client field
  //   expect(getProcessingMockFn).not.toHaveBeenCalled();
  // });
});
