import React from "react";

import { render, cleanup, wait } from "test-utils";
import TextLoader from ".";
import { Classes } from "@blueprintjs/core";

describe("TextLoader", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("renders without error", async () => {
    render(<TextLoader />);
    await wait();
  });

  it("should shows skeleton loader if text if nullable", async () => {
    const { asFragment, getByText, rerender } = render(
      <TextLoader text={undefined} />,
    );
    expect(document.querySelector(`.${Classes.SKELETON}`)).toBeInTheDocument();
    rerender(<TextLoader loading text={undefined} />);
    expect(document.querySelector(`.${Classes.SKELETON}`)).toBeInTheDocument();
    expect(getByText("_".repeat(5))).toBeInTheDocument();
    expect(asFragment()).toMatchSnapshot();
  });

  it("should shows skeleton loading is true", async () => {
    render(<TextLoader loading text={"text"} />);
    expect(document.querySelector(`.${Classes.SKELETON}`)).toBeInTheDocument();
  });
  it("should shows custom length", async () => {
    const { asFragment, getByText } = render(<TextLoader length={10} />);
    expect(getByText("_".repeat(10))).toBeInTheDocument();
    expect(asFragment()).toMatchSnapshot();
  });
  it("should shows skeleton if loading is true", async () => {
    const { queryByText } = render(<TextLoader loading text={"text"} />);
    expect(document.querySelector(`.${Classes.SKELETON}`)).toBeInTheDocument();
    expect(queryByText("text")).toBeNull();
  });
  it("should shows text if text is not nullable and loading is false", async () => {
    const { asFragment, getByText } = render(<TextLoader text={"text"} />);
    expect(getByText("text")).toBeInTheDocument();
    expect(asFragment()).toMatchSnapshot();
  });
  it("should use span as default", async () => {
    const { getByText, asFragment } = render(<TextLoader text={"text"} />);
    expect(getByText("text").tagName.toLowerCase()).toBe("span");
    expect(asFragment()).toMatchSnapshot();
  });
  it("should use div wrapper with fill option", async () => {
    const { getByText, asFragment } = render(<TextLoader fill text={"text"} />);
    expect(getByText("text").tagName.toLowerCase()).toBe("div");
    expect(asFragment()).toMatchSnapshot();
  });
  it("should use custom wrapper tag", async () => {
    const { getByText, asFragment } = render(
      <TextLoader wrapperTagName="p" text={"text"} />,
    );
    expect(getByText("text").tagName.toLowerCase()).toBe("p");
    expect(asFragment()).toMatchSnapshot();
  });
  // TODO test ellipsize
});
