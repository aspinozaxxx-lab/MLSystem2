import React from "react";
import { render, cleanup } from "@testing-library/react";

import { setTestId } from "./set-testid";

describe("set-testid", () => {
  afterEach(cleanup);

  it("should set data-testid", () => {
    const { getByTestId } = render(<div ref={setTestId`my-id`} />);
    expect(getByTestId("my-id")).toBeInTheDocument();
  });
});
