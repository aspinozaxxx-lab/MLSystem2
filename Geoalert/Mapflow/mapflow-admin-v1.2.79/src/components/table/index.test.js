import React from "react";
import { IconNames } from "@blueprintjs/icons";

import { render, cleanup, fireEvent } from "test-utils";
import Table from ".";

import data from "example-table-data";
jest.mock("example-table-data");

const mapNodes = (nodes, getValue) => {
  const result = [];
  for (const [, value] of nodes.entries()) result.push(getValue(value));
  return result;
};

describe("Table", () => {
  let component, columns;
  beforeEach(() => {
    columns = [
      {
        Header: "id",
        id: "id",
        accessor: "id",
      },
      {
        Header: "Name",
        id: "name",
        accessor: "name",
      },
      {
        Header: "Description",
        id: "description",
        accessor: "description",
      },
      {
        Header: "Static",
        id: "static",
        Cell: () => "this is static",
        disableSortBy: true,
      },
    ];
    component = <Table data={data} columns={columns} />;
  });
  afterEach(cleanup);

  it("renders without error", () => {
    render(component);
  });

  it("matches snapshot", () => {
    const { asFragment } = render(component);
    expect(asFragment()).toMatchSnapshot();
  });

  it("should handle row click", () => {
    const onRowClick = jest.fn();
    const { getByText } = render(
      <Table data={data} columns={columns} onRowClick={onRowClick} />,
    );
    fireEvent.click(getByText(data[0]["name"]));
    expect(onRowClick).toHaveBeenCalledWith(data[0]);
  });

  it("should render headers and data", () => {
    const { getByText, getAllByText } = render(component);
    for (const { Header } of columns)
      expect(getByText(Header)).toBeInTheDocument();
    for (const { id, name, description } of data) {
      expect(getByText(id)).toBeInTheDocument();
      expect(getByText(name)).toBeInTheDocument();
      expect(getByText(description)).toBeInTheDocument();
    }
    expect(getAllByText("this is static")).toHaveLength(3);
    expect(getAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(3);
  });

  it("should sort", () => {
    const { getByText, getAllByText, queryAllByText } = render(component);
    // initially sorted asc by first column
    let original = data.map(({ id }) => id).sort();
    let rows = getAllByText(/name/);
    expect(mapNodes(rows, (v) => v.parentNode.firstChild.textContent)).toEqual(
      original,
    );
    expect(queryAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(3);

    // sort asc by name
    fireEvent.click(getByText("Name"));
    expect(queryAllByText(IconNames.CARET_UP)).toHaveLength(1);
    expect(queryAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(0);
    original = data.map(({ name }) => name).sort();
    rows = getAllByText(/name/);
    expect(mapNodes(rows, (v) => v.textContent)).toEqual(original);

    // sort desc by name
    fireEvent.click(getByText("Name"));
    expect(queryAllByText(IconNames.CARET_DOWN)).toHaveLength(1);
    expect(queryAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(0);
    rows = getAllByText(/name/);
    expect(mapNodes(rows, (v) => v.textContent)).toEqual(original.reverse());

    // switch off sorting
    fireEvent.click(getByText("Name"));
    expect(queryAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(3);
    original = data.map(({ id }) => id).sort();
    rows = getAllByText(/name/);
    expect(mapNodes(rows, (v) => v.parentNode.firstChild.textContent)).toEqual(
      original,
    );

    // sort asc by description
    fireEvent.click(getByText("Description"));
    expect(queryAllByText(IconNames.CARET_UP)).toHaveLength(1);
    expect(queryAllByText(IconNames.DOUBLE_CARET_VERTICAL)).toHaveLength(0);
    original = data.map(({ description }) => description).sort();
    rows = getAllByText(/description/);
    expect(mapNodes(rows, (v) => v.textContent)).toEqual(original);
  });
});
