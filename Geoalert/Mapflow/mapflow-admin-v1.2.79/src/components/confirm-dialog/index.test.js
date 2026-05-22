import React from "react";

import {
  render,
  cleanup,
  wait,
  waitForElement,
  waitForElementToBeRemoved,
  fireEvent,
  act,
} from "test-utils";
import ConfirmDialog from ".";

describe("ConfirmDialog", () => {
  let component, closeFn, confirmMock;
  beforeEach(() => {
    confirmMock = jest.fn((close) => (closeFn = close));
    component = (
      <ConfirmDialog
        onConfirm={confirmMock}
        confirmButtonText="Confirm"
        cancelButtonText="Cancel"
        text="Alert message"
      >
        {({ showDialog }) => <button onClick={showDialog}>Open</button>}
      </ConfirmDialog>
    );
  });
  afterEach(cleanup);

  it("should open by child button click", async () => {
    const { getByText, asFragment } = render(component);
    const open = await waitForElement(() => getByText("Open"));
    fireEvent.click(open);
    await waitForElement(() => getByText("Confirm"));
    await waitForElement(() => getByText("Cancel"));
    await waitForElement(() => getByText("Alert message"));
    expect(asFragment()).toMatchSnapshot();
  });

  it("should close on cancel", async () => {
    const { getByText } = render(component);
    const open = await waitForElement(() => getByText("Open"));
    fireEvent.click(open);
    const cancel = await waitForElement(() => getByText("Cancel"));
    fireEvent.click(cancel);
    await waitForElementToBeRemoved(() => getByText("Cancel"));
    await wait();
  });

  it("should call onConfirm on confirm and returns close fn", async () => {
    const { getByText } = render(component);
    const open = await waitForElement(() => getByText("Open"));
    fireEvent.click(open);
    const confirm = await waitForElement(() => getByText("Confirm"));
    fireEvent.click(confirm);
    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(confirmMock).toHaveReturnedWith(closeFn);
    act(() => {
      closeFn();
    });
    await waitForElementToBeRemoved(() => getByText("Confirm"));
    await wait();
  });
});
