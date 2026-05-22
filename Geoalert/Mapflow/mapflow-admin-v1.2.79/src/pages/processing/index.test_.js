import React from "react";
import { QueryCache } from "@tanstack/react-query";

import {
  render,
  cleanup,
  wait,
  within,
  fireEvent,
  waitForElement,
  waitForElementToBeRemoved,
} from "test-utils";
import processing from "fixtures/processing";
import {
  getSpinner,
  mockGetBoundingClientRect,
  mockOffsetSize,
} from "test-utils/helpers";

import { PROCESSING } from "constants/routes";
import Processing from ".";

import { GET_PROJECT_NAME } from "containers/breadcrumbs";
import { GET_PROCESSING } from "containers/processing-sidebar";
import projects from "fixtures/projects";
import { IconNames } from "@blueprintjs/icons";

jest.mock("constants/envs");

const [project] = projects;

describe("Processing page", () => {
  let component, options;
  const projectName = project["name"];
  const processingDesc = processing["description"];
  const processingName = processing["name"];
  beforeEach(() => {
    mockOffsetSize(200, 300);
    mockGetBoundingClientRect();
    const processingId = processing["id"];
    const projectId = "1";
    const path = PROCESSING;
    const processingRoute = `/projects/${projectId}/processings/${processingId}`;
    const mocks = [
      {
        request: { query: GET_PROCESSING, variables: { processingId } },
        result: { data: { processing } },
      },
      {
        request: { query: GET_PROCESSING, variables: { processingId } },
        result: { data: { processing } },
      },
      {
        request: { query: GET_PROJECT_NAME, variables: { id: projectId } },
        result: { data: { project: { name: projectName } } },
      },
    ];

    QueryCache.setQueryData(["processing", processingId], {
      ...processing,
      bbox: processing.bbox,
    });

    component = <Processing />;
    options = {
      withApollo: { mocks },
      withRouter: { route: processingRoute, path },
    };
  });
  afterEach(cleanup);

  // it("should render without error", async () => {
  //   const { getByText, asFragment } = render(component, options);

  //   // check the map loading messages
  //   expect(getSpinner()).toBeInTheDocument();
  //   // await waitForElement(() => getByText("Fetching the processing"));
  //   await waitForElementToBeRemoved(() => getByText("Initialize the map"));
  //   await waitForElement(() => document.querySelector(`#mapbox-map`));

  //   // check the breadcrumb names
  //   // const breadcrumbsParent = document.querySelector(".breadcrumbs");
  //   // await waitForElement(() =>
  //   //   within(breadcrumbsParent).getByText(processingName),
  //   // );
  //   // await waitForElement(() =>
  //   //   within(breadcrumbsParent).getByText(projectName),
  //   // );

  //   expect(asFragment()).toMatchSnapshot();
  //   await wait();
  // });

  // it("should show upload aoi dialog", async () => {
  //   const { getByTestId, getByText } = render(component, options);
  //   const getUploadDialog = () => document.querySelector(".upload-aoi-dialog");

  //   const uploadButton = await waitForElement(() =>
  //     getByTestId("empty-message-upload-button"),
  //   );
  //   fireEvent.click(uploadButton);
  //   await waitForElement(() => getByText(/Upload files/i));
  //   expect(getUploadDialog()).toBeInTheDocument();

  //   // will closed by cross button
  //   fireEvent.click(within(getUploadDialog()).getByText(IconNames.SMALL_CROSS));
  //   await waitForElementToBeRemoved(() => getUploadDialog());
  // });
});
