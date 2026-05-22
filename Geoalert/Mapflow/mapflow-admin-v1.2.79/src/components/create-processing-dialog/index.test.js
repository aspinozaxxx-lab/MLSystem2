import React from "react";

import {
  render,
  cleanup,
  fireEvent,
  wait,
  waitForElementToBeRemoved,
} from "test-utils";

import CreateProcessingDialog, { fieldsConfig } from ".";

// jest.mock("processing-list-mock");

const Errors = {
  REQUIRED: /This field is required/i,
  TO_LONG_NAME: `The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  TO_LONG_DESC: `The description field may not be greater than ${fieldsConfig.desc.maxLength} characters`,
};

describe("CreateProcessingDialog", () => {
  let component, workflowDefId, workflowName, workflowDesc;
  beforeEach(() => {
    workflowName = "Wname";
    workflowDesc = "Wdesc";
    workflowDefId = "2";
    component = (
      <CreateProcessingDialog
        defaultIsOpen
        workflowDefId={workflowDefId}
        workflowName={workflowName}
        workflowDesc={workflowDesc}
      />
    );
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

  it("should close on cancel", async () => {
    const { getByTestId } = render(component);
    await wait();
    fireEvent.click(getByTestId("cancel-create-processing"));
    await waitForElementToBeRemoved(() =>
      getByTestId("cancel-create-processing"),
    );
  });

  it("should validate form fields", async () => {
    const { findByText, getByTestId, getByLabelText } = render(component);
    fireEvent.click(getByTestId("submit-create-processing"));
    let errorMessage = await findByText(Errors.REQUIRED);
    const name = getByLabelText(/Name/i);
    fireEvent.change(name, {
      target: { value: "a".repeat(fieldsConfig.name.maxLength + 1) },
    });
    fireEvent.click(getByTestId("submit-create-processing"));
    errorMessage = await findByText(Errors.TO_LONG_NAME);
    expect(errorMessage).toBeInTheDocument();

    const description = getByLabelText(/Description/i);
    fireEvent.change(description, {
      target: { value: "a".repeat(fieldsConfig.desc.maxLength + 1) },
    });
    fireEvent.click(getByTestId("submit-create-processing"));
    errorMessage = await findByText(Errors.TO_LONG_DESC);
    expect(errorMessage).toBeInTheDocument();
  });
});
