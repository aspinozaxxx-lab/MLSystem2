import React, { useMemo, useState } from "react";
import { Trans } from "@lingui/react";
import { Tag, Icon } from "@blueprintjs/core";
import { gql, useApolloClient } from "@apollo/client";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import stable from "stable";
import pathOr from "ramda/src/pathOr";

import {
  EmptyMessage,
  ErrorMessage,
  Table,
  StateLoading,
  FormatArea,
} from "components";
import { useGoTo } from "hooks/use-go-to";
import { PROJECT_PROCESSING } from "constants/routes";
import { POLL_INTERVAL } from "constants/envs";
import Actions from "./actions";

import { StatusTag } from "components";
import { sortByDate } from "utils/sort-by-date";
import AoiMessagesDialog from "../../containers/aoi-list/aoi-messages-dialog";
import { getProcessETAWithDate } from "utils/getProcessETA";
import ToLocaleTime from "components/toLocalTime";
import { STATUS_SUCCESS } from "constants/common";

export const GET_PROCESSINGS = gql`
  query getProcessings($projectId: ID!) {
    processings(projectIds: [$projectId]) {
      id
      name
      created
      updated
      description
      area
      dataProvider {
        id
        displayName
      }
      progress {
        status
        percentCompleted
        estimate
        completionDate
        details {
          count
          status
          area
          statusUpdateDate
        }
      }
      messages {
        message
      }
      workflowDef {
        name
      }
    }
  }
`;

const getProcessings = (client) => async (projectId) => {
  const result = await client.query({
    query: GET_PROCESSINGS,
    fetchPolicy: "no-cache",
    variables: { projectId },
  });
  return result?.data?.processings;
};

function renderActions({ cell }) {
  const { id, name } = cell.row.original;
  return <Actions id={id} name={name} />;
}

function ProcessingList() {
  const [errorModalOpen, setErrorModalOpen] = useState(false);
  const [messages, setMessages] = useState([]);

  const handleClose = () => {
    setMessages((_) => []);
    setErrorModalOpen(false);
  };

  const { projectId } = useParams();

  const goTo = useGoTo(PROJECT_PROCESSING, { projectId });
  const goToProcessing = ({ id }) => goTo({ processingId: id });

  const client = useApolloClient();

  const { data, status } = useQuery({
    queryKey: ["processings", projectId],
    queryFn: () => getProcessings(client)(projectId),
    refetchInterval: POLL_INTERVAL,
  });

  const processings = useMemo(() => stable(data || [], sortByDate), [data]);
  const columns = [
    {
      Header: (
        <>
          <Trans id="Workflow" />
        </>
      ),
      accessor: "workflowDef.name",
      Cell: ({ cell }) => (
        <Tag>
          {" "}
          <span style={{ marginRight: "5px" }}>
            <Icon icon="application" />
          </span>
          {cell.value}
        </Tag>
      ),
    },
    {
      Header: <Trans id="Name" />,
      id: "name",
      accessor: "name",
    },
    {
      Header: <Trans id="Area" />,
      id: "area",
      accessor: "area",
      Cell: ({ cell }) => <FormatArea cutZeros area={cell.value} />,
    },
    {
      Header: <Trans id="Created" />,
      id: "created",
      accessor: "created",
      Cell: ({ cell }) => {
        return <ToLocaleTime time={cell.value} />;
      },
    },
    {
      Header: <Trans id="Data Provider" />,
      id: "data-provider",
      accessor: "dataProvider.displayName",
      Cell: ({ cell }) => cell.value ?? "—",
    },
    {
      Header: (
        <>
          {" "}
          <Trans id="Status" /> / <Trans id="Progress" />{" "}
        </>
      ),
      accessor: "progress.status",
      Cell: ({ cell }) => {
        const messages = pathOr(null, ["messages"])(cell.row.original);
        const a = cell.row.original;
        return (
          <div className="progress-status">
            <StatusTag
              statusCode={cell.value}
              percent={a.progress.percentCompleted}
              withPercent
            />
            {messages && messages.length > 0 && (
              <Icon
                icon="info-sign"
                className="aoi-messages-icon"
                onClick={(event) => {
                  event.stopPropagation();
                  setMessages(messages);
                  if (messages.length > 0) {
                    setErrorModalOpen(true);
                  }
                }}
              />
            )}
          </div>
        );
      },
    },
    {
      Header: <Trans id="ETA" />,
      accessor: "ETAID",
      disableSortBy: true,
      Cell: ({ cell }) => {
        const estimate = cell?.row.original.progress.estimate;

        if (cell?.row.original.progress.status === STATUS_SUCCESS) {
          return (
            <Tag minimal round intent={"success"}>
              <ToLocaleTime time={cell?.row.original.progress.completionDate} />
            </Tag>
          );
        }

        return estimate > 0 && estimate ? (
          <Tag minimal round intent={"warning"}>
            {getProcessETAWithDate(estimate)}
          </Tag>
        ) : (
          <Trans id="No data" />
        );
      },
    },
    {
      Header: <Trans id="Actions" />,
      accessor: "id",
      Cell: renderActions,
      disableSortBy: true,
    },
  ];

  if (status === "error")
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Processings was not fetched" />}
      />
    );

  if (status === "loading")
    return <StateLoading title={<Trans id="Fetching Processings" />} />;

  if (processings.length === 0)
    return (
      <EmptyMessage
        title={<Trans id="You haven’t created any processing yet" />}
      />
    );

  return (
    <>
      <Table
        striped
        interactive
        data={processings}
        columns={columns}
        onRowClick={goToProcessing}
      />
      <AoiMessagesDialog
        handleClose={handleClose}
        isOpen={errorModalOpen && messages.length > 0}
        messages={messages}
      />

    </>
  );
}

export default React.memo(ProcessingList);
