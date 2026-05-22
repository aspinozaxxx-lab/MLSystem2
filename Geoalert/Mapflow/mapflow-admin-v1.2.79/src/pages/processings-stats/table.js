import { Icon, Tag } from "@blueprintjs/core";
import { Trans } from "@lingui/macro";
import Table from "components/table";
import { useMemo } from "react";
import StatusTag from "components/status-tag";
import AoiMessagesDialog from "containers/aoi-list/aoi-messages-dialog";
import ToLocaleTime from "components/toLocalTime";
import { STATUS_PENDING, STATUS_SUCCESS } from "constants/common";
import { getProcessETAWithCurrentDate } from "utils/getProcessETA";

export const ProcessingsTable = ({ data, sort, handleHeaderClick }) => {
  const columns = useMemo(() => {
    return [
      {
        Header: (
          <>
            <Trans id="Workflow" />
          </>
        ),
        accessor: "workflowDef.name",
        Cell: ({ cell }) => (
          <Tag>
            <span style={{ marginRight: "5px" }}>
              <Icon icon="application" />
            </span>
            {cell.value}
          </Tag>
        ),
        sortDirection:
          sort.accessor === "workflowDef.name" ? sort.sortBy : "none",
        order: "SCENARIO",
      },
      {
        Header: <Trans id="Name" />,
        id: "name",
        accessor: "name",
        sortDirection: sort.accessor === "name" ? sort.sortBy : "none",
        order: "NAME",
      },
      {
        Header: <Trans id="Project name" />,
        id: "projectName",
        accessor: "projectName",
        sortDirection: sort.accessor === "projectName" ? sort.sortBy : "none",
        order: "PROJECT_NAME",
      },
      {
        Header: "Пользователь",
        id: "email",
        accessor: "email",
        sortDirection: sort.accessor === "email" ? sort.sortBy : "none",
        order: "EMAIL",
        Cell: ({ cell }) => (
          <>
            {cell?.row.original.user.name ||
              cell?.row.original.user.preferredUsername ||
              cell?.row.original.user.email}
          </>
        ),
      },
      {
        Header: <Trans id="Status" />,
        accessor: "progress.status",
        Cell: ({ cell }) => {
          const messages = cell.row.original?.messages ?? null;
          return (
            <div className="progress-status">
              <StatusTag statusCode={cell.value} />
              <AoiMessagesDialog messages={messages} />
            </div>
          );
        },
        sortDirection:
          sort.accessor === "progress.status" ? sort.sortBy : "none",
        order: "STATUS",
      },
      {
        Header: <Trans id="Created" />,
        id: "created",
        accessor: "created",
        Cell: ({ cell }) => {
          return <ToLocaleTime time={cell.value} />;
        },
        sortDirection: sort.accessor === "created" ? sort.sortBy : "none",
        order: "CREATED",
      },
      {
        Header: <Trans id="completionDate" />,
        id: "completionDate",
        accessor: "progress.completionDate",
        Cell: ({ cell }) => {
          if (cell.row.original.progress.status === STATUS_SUCCESS) {
            return (
              <Tag minimal round intent={"success"}>
                <ToLocaleTime time={cell.value} />
              </Tag>
            );
          }

          if (cell.row.original.progress.status === STATUS_PENDING) {
            if (
              cell.row.original.progress.estimate === 0 ||
              !cell.row.original.progress.estimate
            ) {
              return "Нет данных";
            }

            return (
              <Tag minimal round intent={"warning"}>
                {getProcessETAWithCurrentDate(
                  cell.row.original.progress.estimate,
                )}
              </Tag>
            );
          }

          if (!cell.value) {
            return "Нет данных";
          }
        },
        sortDirection:
          sort.accessor === "progress.completionDate" ? sort.sortBy : "none",
        order: "COMPLETED",
      },
      {
        Header: <Trans id="Progress" />,
        // accessor: "progress.percentCompleted",
        id: "progress",
        Cell: ({ row }) => `${row.original.progress.percentCompleted} %`,
      },
    ];
  }, [sort]);

  return (
    <Table
      striped
      data={data}
      columns={columns}
      sort={sort}
      handleHeaderClick={handleHeaderClick}
    />
  );
};

export default ProcessingsTable;
