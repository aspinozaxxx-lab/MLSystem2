import React from "react";
import {
  render,
  waitForElement,
  waitForElementToBeRemoved,
  cleanup,
  fireEvent,
  wait,
} from "test-utils";
import projects from "fixtures/projects";

import Projects from ".";
import { responseWithDelay } from "test-utils/response-with-delay";
import client from "graphql/client";
jest.mock("graphql/client");

describe("Projects page", () => {
  beforeEach(() => {});
  afterEach(() => {
    jest.clearAllMocks();
    cleanup();
  });

  // it("should show loader", async () => {
  //   client.query.mockImplementationOnce(() => {});
  //   const { getByText, asFragment } = render(<Projects />);
  //   expect(asFragment()).toMatchSnapshot();
  //   await waitForElement(() => getByText(/Fetching Projects/i));
  //   await wait();
  // });

  // it("should show empty message", async () => {
  //   client.query.mockImplementationOnce(responseWithDelay([]));
  //   const { getByText, asFragment, getByTestId } = render(<Projects />);
  //   // await waitForElementToBeRemoved(() => getByText(/Fetching Projects/i));
  //   await waitForElement(() =>
  //     getByText(/You haven’t created any project yet/i),
  //   );
  //   expect(asFragment()).toMatchSnapshot();
  //   fireEvent.click(getByTestId("create-new-project"));
  //   await waitForElement(() => getByText("Project creation"));
  // });

  // it("should show projects", async () => {
  //   client.query.mockImplementationOnce(
  //     responseWithDelay({ data: { projects } }),
  //   );
  //   const { getByText, asFragment } = render(<Projects />);
  //   await wait();
  //   for (const project of projects)
  //     await waitForElement(() => getByText(project.name));
  //   expect(asFragment()).toMatchSnapshot();
  //   await wait();
  // });

  it("should show error message", async () => {
    client.query.mockImplementationOnce(
      responseWithDelay(null, { error: "Error" }),
      // responseWithDelay(null, { error: "Error", delay: 1 }),
    );
    const { getByText, asFragment } = render(<Projects />);
    await waitForElement(() => getByText(/Error fetch projects/i));
    expect(asFragment()).toMatchSnapshot();
    await wait();
  });
});
