import React from "react";

import { render, cleanup, wait, fireEvent, waitForElement } from "test-utils";

import { CreateProjectDialog, fieldsConfig } from ".";

const Errors = {
  REQUIRED: /This field is required/i,
  TO_LONG_NAME: `The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  TO_LONG_DESC: `The description field may not be greater than ${fieldsConfig.description.maxLength} characters`,
};

describe("CreateProjectDialog", () => {
  let component, handleClose, createProjectMock;
  beforeEach(() => {
    handleClose = jest.fn();
    createProjectMock = jest.fn();
    component = (
      <CreateProjectDialog
        isOpen={true}
        handleClose={handleClose}
        createProject={createProjectMock}
      />
    );
  });
  afterEach(() => {
    jest.clearAllMocks();
    cleanup();
  });

  it("renders without error", async () => {
    render(component);
    await wait();
  });

  it("matches snapshot", async () => {
    const { asFragment } = render(component);
    await wait();
    expect(asFragment()).toMatchSnapshot();
  });

  it("should close on cancel", async () => {
    const { getByTestId } = render(component);
    const cancelButton = await waitForElement(() =>
      getByTestId("cancel-create-project"),
    );
    fireEvent.click(cancelButton);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("should validate form fields", async () => {
    const { findByText, getByTestId } = render(component);
    const submitButton = await waitForElement(() =>
      getByTestId("submit-create-project"),
    );
    fireEvent.click(submitButton);
    let errorMessage = await findByText(Errors.REQUIRED);
    const name = getByTestId("project-name");
    fireEvent.change(name, {
      target: { value: "a".repeat(fieldsConfig.name.maxLength + 1) },
    });
    fireEvent.click(submitButton);
    errorMessage = await findByText(Errors.TO_LONG_NAME);
    expect(errorMessage).toBeInTheDocument();

    const description = getByTestId("project-description");
    fireEvent.change(description, {
      target: { value: "a".repeat(fieldsConfig.description.maxLength + 1) },
    });
    fireEvent.click(submitButton);
    errorMessage = await findByText(Errors.TO_LONG_DESC);
    expect(errorMessage).toBeInTheDocument();
  });

  it("should call create project on submit", async () => {
    const { getByTestId } = render(component);
    const nameInput = await waitForElement(() => getByTestId("project-name"));
    fireEvent.change(nameInput, { target: { value: "New Project" } });
    fireEvent.click(getByTestId("submit-create-project"));
    await wait();
    expect(createProjectMock).toHaveBeenCalledTimes(1);
  });

  it("should disable controls on loading", async () => {
    const { getByTestId } = render(
      <CreateProjectDialog
        isLoading
        isOpen={true}
        handleClose={handleClose}
        createProject={createProjectMock}
      />,
    );
    const nameInput = await waitForElement(() => getByTestId("project-name"));
    const descriptionInput = getByTestId("project-description");
    const subminButton = getByTestId("submit-create-project");
    const cancelButton = getByTestId("cancel-create-project");
    expect(subminButton).toBeDisabled();
    expect(cancelButton).toBeDisabled();
    expect(nameInput).toBeDisabled();
    expect(descriptionInput).toBeDisabled();
    await wait();
  });
});
