import React, { useMemo, useState } from "react";
import { useTable, useSortBy } from "react-table";
import classnames from "classnames";
import { HTMLTable, Icon, Classes } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { useTheme } from "hooks/use-theme";

function getSortinIcon(column, sortBy) {
  if (column.order && column.sortDirection) {
    if (column.sortDirection === "ASC") return IconNames.CARET_UP;
    if (column.sortDirection === "DESC") return IconNames.CARET_DOWN;
    return IconNames.DOUBLE_CARET_VERTICAL;
  }

  if (column.isSorted)
    if (column.isSortedDesc) return IconNames.CARET_DOWN;
    else return IconNames.CARET_UP;
  return IconNames.DOUBLE_CARET_VERTICAL;
}
export function HeaderColumn({ column, sortBy, filter, handleHeaderClick }) {
  const iconName = getSortinIcon(column, sortBy);
  const iconClasses = classnames("sorting-icon", {
    "sorting-icon--selected": column.isSorted,
  });
  const headerCellProps = column.getHeaderProps(column.getSortByToggleProps());

  const style = column.disableSortBy ? { pointerEvents: "none" } : {};
  return (
    <th
      className="header"
      {...headerCellProps}
      style={style}
      {...(handleHeaderClick
        ? { onClick: () => handleHeaderClick(column) }
        : {})}
    >
      <div className="header-cell">
        {column.render("Header")}
        <span className={iconClasses}>
          {column.canSort && <Icon icon={iconName} />}
        </span>
      </div>
    </th>
  );
}

function Table({
  data,
  columns,
  className,
  bodyClassName,
  headingClassName,
  style,
  condensed = false,
  striped = true,
  bordered = false,
  interactive = false,
  onRowClick = () => {},
  indexColumn = false,
  disableSortByIndex = true,
  indexHeader = "#",
  // loading,
  // loadingRowsCount = 5,
  handleHeaderClick,
}) {
  const { themeClassName } = useTheme();

  const columnsConfig = useMemo(
    () =>
      indexColumn
        ? [
            {
              id: "data-table-row-index",
              accessor: "index",
              Header: indexHeader,
              defaultCanSort: !disableSortByIndex,
              Cell: ({ row }) => row.index + 1,
              disableSortBy: disableSortByIndex,
            },
            ...columns,
          ]
        : columns,
    [columns, indexHeader, indexColumn, disableSortByIndex],
  );

  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
    state,
  } = useTable(
    {
      columns: columnsConfig,
      data,
      // manualSorting: false,
    },
    useSortBy,
  );

  return (
    <HTMLTable
      {...getTableProps()}
      condensed={condensed}
      interactive={interactive}
      striped={striped}
      // bordered={bordered}
      className={classnames("table", Classes.ELEVATION_0, className, themeClassName)}
      style={style}
    >
      <thead className={classnames("table__head", headingClassName)}>
        {headerGroups.map((headerGroup, i) => (
          <tr key={i} {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map((column, i) => (
              <HeaderColumn
                key={i}
                sortBy={state.sortBy}
                column={column}
                handleHeaderClick={handleHeaderClick}
              />
            ))}
          </tr>
        ))}
      </thead>
      <tbody
        className={classnames("table__body", bodyClassName)}
        {...getTableBodyProps()}
      >
        {rows.map((row, i) => {
          prepareRow(row);
          return (
            <tr
              key={i}
              {...row.getRowProps()}
              onClick={() => onRowClick(row.original)}
            >
              {row.cells.map((cell, i) => {
                return (
                  <td key={i} {...cell.getCellProps()}>
                    {cell.render("Cell")}
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </HTMLTable>
  );
}

export default React.memo(Table);
