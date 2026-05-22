import React from "react";
import { createMemoryHistory } from "history";

import {
  render,
  cleanup,
  wait,
  fireEvent,
  within,
  waitForElement,
  waitForElementToBeRemoved,
} from "test-utils";
import { getSpinner } from "test-utils/helpers";

import { CREATE_PROCESSING } from "components/create-processing-dialog";
import WorkflowList, { GET_WORKFLOWS } from ".";

import workflows from "fixtures/workflows";

describe("WorkflowList", () => {
  let component;
  beforeEach(() => {
    component = <WorkflowList />;
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

  it("should create processing", async () => {
    const workflowName = workflows["project"]["workflowDefs"][0]["name"];
    const workflowDescription =
      workflows["project"]["workflowDefs"][0]["description"];
    const workflowDefId = workflows["project"]["workflowDefs"][0]["id"];
    const workflowDefs = workflows["project"]["workflowDefs"];

    const createProcessingVariables = {
      projectId: "1",
      name: "processing name",
      description: "processing description",
      workflowDefId,
    };
    const { name, description, projectId } = createProcessingVariables;
    const createdProcessingId = "1";

    const mocks = [
      {
        request: {
          query: GET_WORKFLOWS,
          variables: { projectId },
        },
        result: { data: workflows },
      },
      {
        request: {
          query: CREATE_PROCESSING,
          variables: { data: createProcessingVariables },
        },
        result: {
          data: { createProcessing: { id: createdProcessingId, name } },
        },
      },
      {
        request: {
          query: CREATE_PROCESSING,
          variables: { data: createProcessingVariables },
        },
        error: new Error("Error creating processing"),
      },
    ];

    const path = `/projects/:projectId/*`;
    const workflowsRoute = `/projects/${projectId}/workflows`;
    const history = createMemoryHistory({ initialEntries: [workflowsRoute] });
    const options = {
      withApollo: { mocks },
      withRouter: { history, route: workflowsRoute, path },
    };
    const { findByText, getByText, getByLabelText, getByTestId } = render(
      component,
      options,
    );
    // all workflows rendered
    for (const { name } of workflowDefs)
      await waitForElement(() => getByText(name));

    // create processing from first workflow in the list
    let first = await waitForElement(() => getByText(workflowName));
    const createButton = within(first.parentNode).getByTestId(
      "create-processing",
    );

    fireEvent.click(createButton);
    await waitForElement(() => getByTestId("submit-create-processing"));
    // check workflow info matches selected workflow
    let workflowInfo = document.querySelector(".workflow-info");
    within(workflowInfo).getByText(workflowName);
    within(workflowInfo).getByText(workflowDescription);

    // fill and submit the form
    fireEvent.change(getByLabelText(/Name/), { target: { value: name } });
    fireEvent.change(getByLabelText(/Description/), {
      target: { value: description },
    });
    fireEvent.click(getByTestId("submit-create-processing"));

    await waitForElement(() =>
      findByText(`Processing ${name} successfully created`),
    );
    // if successfull then push to processing page
    expect(history.location.pathname).toBe(
      `/projects/${projectId}/processings/${createdProcessingId}`,
    );

    // create processing with error
    // set prev location
    history.push(workflowsRoute);
    fireEvent.click(createButton);
    await waitForElement(() => getByTestId("submit-create-processing"));
    fireEvent.change(getByLabelText(/Name/), { target: { value: name } });
    fireEvent.change(getByLabelText(/Description/), {
      target: { value: description },
    });
    fireEvent.click(getByTestId("submit-create-processing"));
    await waitForElement(() => findByText(`Error creating processing`));
    // not successfull stay same page
    expect(history.location.pathname).toBe(workflowsRoute);
  }, 10000);
});
