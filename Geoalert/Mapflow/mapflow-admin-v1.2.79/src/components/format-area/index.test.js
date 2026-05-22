import React from "react";

import { render, cleanup } from "test-utils";
import FormatArea, { formatArea } from ".";

describe("formatArea", function () {
  it("should default 2 demicals", function () {
    const [a] = formatArea(1);
    const demicals = a.split(".")[1];
    expect(demicals.length).toBe(2);
  });
  it("should set demicals", function () {
    const [a] = formatArea(1234567, 3);
    const demicals = a.split(".")[1];
    expect(demicals.length).toBe(3);
  });
  it("should format to meters", function () {
    const [a, u] = formatArea(100);
    expect(a).toBe("100.00");
    expect(u).toBe("meters");
  });
  it("should format to kilometers", function () {
    const [a, u] = formatArea(1000000);
    expect(a).toBe("1.00");
    expect(u).toBe("kilometers");
  });
});

describe("FormatArea", () => {
  beforeEach(() => {});
  afterEach(cleanup);

  it("matches snapshot", () => {
    const { asFragment } = render(<FormatArea area={1} />);
    expect(asFragment()).toMatchSnapshot();
  });

  it("should format area", () => {
    const { getByText } = render(<FormatArea area={1} />);
    expect(getByText("1.00 m")).toBeInTheDocument();
  });

  it("should set demicals", () => {
    const { getByText } = render(<FormatArea area={1} demicals={3} />);
    expect(getByText("1.000 m")).toBeInTheDocument();
  });

  it("should convert to km", () => {
    const { getByText } = render(<FormatArea area={1000000} />);
    expect(getByText("1.00 km")).toBeInTheDocument();
  });

  it("should cut tail zeroes", () => {
    const { getByText, rerender } = render(
      <FormatArea cutZeros area={1000000} />,
    );
    expect(getByText("1 km")).toBeInTheDocument();
    rerender(<FormatArea area={1010000} />);
    expect(getByText("1.01 km")).toBeInTheDocument();
  });
});
