import React from "react";

import { render, cleanup, wait, waitForElement, fireEvent } from "test-utils";
import { mockOffsetSize, getSpinnerClass } from "test-utils/helpers";
import { AppToaster } from "toaster";

import { invokeDownload, getDownloadLink } from "./download-by-url";

import AoiListActions, { RUN_PROCESSING } from ".";

import { DELETE_AOIS, GET_PROCESSING_RESULT } from "./actions-menu";

jest.mock("./download-by-url");

describe("AoiListActions", () => {
  const projectId = "1";
  const processingId = "1";

  function getRouterOptions() {
    const path = `/projects/:projectId/processings/:processingId`;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;
    return { route: processingRoute, path };
  }

  function getOptions({ mocks = [] } = {}) {
    return { withApollo: { mocks }, withRouter: getRouterOptions() };
  }

  function getMocks({
    deleteAois = 0,
    createResult = 1,
    runProcessing = 0,
  } = {}) {
    return [
      {
        request: {
          query: RUN_PROCESSING,
          variables: {
            filter: { processingIds: [processingId] },
          },
        },
        result: {
          data: { runProcessing },
        },
      },
      {
        request: {
          query: GET_PROCESSING_RESULT,
          variables: {
            filter: { processingIds: [processingId] },
          },
        },
        result: {
          data: { createResult },
        },
      },
      {
        request: {
          query: DELETE_AOIS,
          variables: {
            filter: { processingIds: [processingId] },
          },
        },
        result: {
          data: { deleteAois },
        },
      },
    ];
  }

  beforeEach(() => {
    mockOffsetSize(200, 300);
    AppToaster.clear();
  });
  afterEach(cleanup);

  it("renders without error", async () => {
    render(<AoiListActions />);
    await wait();
  });

  it("should matches snapshot", async () => {
    const { asFragment } = render(<AoiListActions />);
    await wait();
    expect(asFragment()).toMatchSnapshot();
  });

  it("should run processing", async () => {
    const mocks = getMocks({ runProcessing: 1 });
    const options = getOptions({ mocks });

    const { getByTestId, getByText } = render(<AoiListActions />, options);

    const runButton = await waitForElement(() =>
      getByTestId("run-processing-all"),
    );
    fireEvent.click(runButton);

    expect(runButton).toBeDisabled();
    await waitForElement(() => getByText(/Successfully started 1 AOIs/i));
  });

  it("should delete all", async () => {
    const mocks = getMocks({ deleteAois: 1 });
    const options = getOptions({ mocks });

    const { getByTestId, getByText } = render(<AoiListActions />, options);

    const showActionsButton = await waitForElement(() =>
      getByTestId("show-processing-actions"),
    );
    fireEvent.click(showActionsButton);
    const deleteAllButton = await waitForElement(() =>
      getByText(/Delete all/i),
    );
    fireEvent.click(deleteAllButton);
    await waitForElement(() => getByText(/Deleted 1 AOIs/i));
  });

  it("should export results", async () => {
    const mocks = getMocks({ createResult: "1" });
    const options = getOptions({ mocks });

    const { getByTestId, getByText } = render(<AoiListActions />, options);

    const showActionsButton = await waitForElement(() =>
      getByTestId("show-processing-actions"),
    );
    fireEvent.click(showActionsButton);
    const exportResultsButton = await waitForElement(() =>
      getByText(/Export results/i),
    );
    fireEvent.click(exportResultsButton);
    await waitForElement(() => getByText(/Successfully prepared export data/i));
    expect(getDownloadLink).toHaveBeenNthCalledWith(1, "1");
    expect(invokeDownload).toHaveBeenNthCalledWith(1, ...["1", "1.geojson"]);
  });
});
