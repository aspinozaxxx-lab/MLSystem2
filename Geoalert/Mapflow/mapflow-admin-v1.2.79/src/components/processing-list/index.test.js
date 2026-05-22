import React from "react";
import { createMemoryHistory } from "history";
import { InMemoryCache } from "apollo-cache-inmemory";

import { render, cleanup, wait, fireEvent, waitForElement } from "test-utils";
import ProcessingList, { GET_PROCESSINGS } from ".";

import processings from "fixtures/processings";
import { ProgressStatuses } from "constants/common";
const { T } = ProgressStatuses;

describe("ProcessingList", () => {
  let component;
  beforeEach(() => {
    component = <ProcessingList />;
  });
  afterEach(cleanup);

  it("render processings", async () => {
    const projectId = "1";
    const mocks = [
      {
        request: {
          query: GET_PROCESSINGS,
          variables: { projectId },
        },
        result: {
          data: { processings: processings, __typename: "ProcessingDefs" },
        },
      },
    ];
    const path = `/projects/:projectId/*`;
    const route = `/projects/${projectId}/processings`;
    const options = {
      withApollo: { mocks },
      withRouter: { route, path },
    };

    const { getByText } = render(component, options);
    for (const processing of processings) {
      const { name, description, workflowDef, progress } = processing;
      await waitForElement(() => getByText(name));
      await waitForElement(() => getByText(description));
      await waitForElement(() => getByText(workflowDef.name));
      await waitForElement(() => getByText(`${progress.percentCompleted} %`));
      await waitForElement(() => getByText(T[progress.status]["id"]));
    }
  });

  it("matches snapshot", async () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });

  it("should open by click", async () => {
    const firstName = processings[0]["name"];
    const processingId = processings[0]["id"];
    const projectId = "1";
    const cache = new InMemoryCache({ addTypename: false });
    const mocks = [
      {
        request: {
          query: GET_PROCESSINGS,
          variables: { projectId },
        },
        result: {
          data: { processings: processings, __typename: "ProcessingDefs" },
        },
      },
    ];

    const path = `/projects/:projectId/*`;
    const processingsRoute = `/projects/${projectId}/processings`;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;
    const history = createMemoryHistory({ initialEntries: [processingsRoute] });
    const options = {
      withApollo: { mocks, cache },
      withRouter: { history, route: processingsRoute, path },
    };
    const { findByText, getByText } = render(component, options);
    // all processings rendered
    for (const { name } of processings) await findByText(name);
    fireEvent.click(getByText(firstName));
    expect(history.location.pathname).toBe(processingRoute);
  });
});
