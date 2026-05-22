import React from "react";
import { createMemoryHistory } from "history";

import {
  render,
  cleanup,
  fireEvent,
  waitForElement,
  waitForElementToBeRemoved,
} from "test-utils";
import ProjectCard from ".";
import projects from "fixtures/projects";
import client from "graphql/client";
import { responseWithDelay } from "test-utils/response-with-delay";
import { QueryCache } from "@tanstack/react-query";
jest.mock("graphql/client");

const [card] = projects;

describe("ProjectCard", () => {
  let component;
  beforeEach(() => (component = <ProjectCard {...card} />));
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });

  it("should show tooltip", async () => {
    const { getByText } = render(component);
    const updatedDate = getByText(/updated/i);
    fireEvent.mouseEnter(updatedDate);
    await waitForElement(() => getByText(/created/i));
    fireEvent.mouseLeave(updatedDate);
    await waitForElementToBeRemoved(() => getByText(/created/i));
  });

  it("should go to the project onclick", async () => {
    const history = createMemoryHistory();
    const { name, id } = card;
    const { getByText } = render(component, { withRouter: { history } });
    fireEvent.click(getByText(name));
    expect(history.location.pathname).toBe(`/projects/${id}/workflows`);
  });

  it("should delete project", async () => {
    QueryCache.setQueryData("projects", [card]);
    client.mutate.mockImplementationOnce(
      responseWithDelay({ data: { deleteProject: card } }),
    );
    const { getByText, getByTestId } = render(component);
    const deleteProject = await waitForElement(() =>
      getByTestId("delete-project"),
    );
    fireEvent.click(deleteProject);
    const confirmDelete = await waitForElement(() => getByText("Delete"));
    fireEvent.click(confirmDelete);
    await waitForElement(() => getByText(`Project "${card.name}" deleted`));
  });

  it("should show error on delete project failed", async () => {
    client.mutate.mockImplementationOnce(
      responseWithDelay(null, { error: "Error" }),
    );
    const { getByText, getByTestId } = render(component);
    const deleteProject = await waitForElement(() =>
      getByTestId("delete-project"),
    );
    fireEvent.click(deleteProject);
    const confirmDelete = await waitForElement(() => getByText("Delete"));
    fireEvent.click(confirmDelete);
    await waitForElement(() => getByText("Error delete project"));
  });
});
