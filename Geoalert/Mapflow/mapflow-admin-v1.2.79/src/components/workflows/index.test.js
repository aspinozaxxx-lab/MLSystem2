import React from "react";
import { createMemoryHistory } from "history";

import { render, cleanup, wait, fireEvent, waitForElement } from "test-utils";

import Workflows from ".";
import workflows from "fixtures/workflows";
import { GET_WORKFLOWS } from "components/workflow-list";

describe("Workflows", () => {
  let component;
  beforeEach(() => {
    component = <Workflows />;
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

  it("should go to create workflow", async () => {
    const projectId = "1";
    const mocks = [
      {
        request: {
          query: GET_WORKFLOWS,
          variables: { projectId },
        },
        result: { data: workflows },
      },
    ];

    const path = `/projects/:projectId/*`;
    const workflowsRoute = `/projects/${projectId}/workflows`;
    const history = createMemoryHistory({ initialEntries: [workflowsRoute] });
    const options = {
      withApollo: { mocks },
      withRouter: { history, route: workflowsRoute, path },
    };
    const { getByTestId, asFragment } = render(component, options);
    const createButton = await waitForElement(() =>
      getByTestId("create-workflow"),
    );
    // expect(asFragment()).toMatchSnapshot();
    fireEvent.click(createButton);
    expect(history.location.pathname).toBe(`${workflowsRoute}/create`);
    await wait();
  });
});
