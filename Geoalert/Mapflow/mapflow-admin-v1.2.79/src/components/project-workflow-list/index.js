import React, { useMemo } from "react";
import { Trans } from "@lingui/macro";
import { useQuery } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";
import { useParams } from "react-router-dom";
import stable from "stable";

import { POLL_INTERVAL } from "constants/envs";
import { EmptyMessage, ErrorMessage, Table, StateLoading } from "components";
import LinkWorkflow from "components/link-workflow-button";

import Actions from "./actions";
import { sortByDate } from "utils/sort-by-date";
import { Tag } from "@blueprintjs/core";
import ToLocaleTime from "components/toLocalTime";

export const GET_PROJECT_WORKFLOWS = gql`
  query getProjectWorkflows($projectId: ID!) {
    project(id: $projectId) {
      workflowDefs {
        id
        name
        created
        description
        isDefault
        blocks {
          name
          displayName
          optional
          defaultEnabled
        }
      }
    }
  }
`;

function ProjectWorkflowList() {
  const { projectId } = useParams();

  const client = useApolloClient();

  const { data, status } = useQuery({
    queryKey: ["projectWorkflows", projectId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_PROJECT_WORKFLOWS,
        fetchPolicy: "no-cache",
        variables: { projectId },
      });

      return result?.data?.project?.workflowDefs;
    },
    refetchInterval: POLL_INTERVAL,
    initialData: [],
  });

  const workflowDefs = useMemo(() => stable(data, sortByDate), [data]);

  const columns = [
    {
      Header: <Trans id="Name" />,
      id: "name",
      accessor: "name",
      Cell: ({ row }) => {
        const { name, isDefault } = row.original;
        return (
          <div className="workflow-cell">
            <span className="workflow-cell__name">{name}</span>
            {isDefault && (
              <Tag round>
                <Trans id="default" />
              </Tag>
            )}
          </div>
        );
      },
    },
    {
      Header: <Trans id="Description" />,
      id: "description",
      accessor: "description",
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
      Header: <Trans id="Actions" />,
      Cell: ({ row }) => {
        const { id, name, description, isDefault, blocks } = row.original;
        return (
          <Actions
            workflowDefId={id}
            projectId={projectId}
            name={name}
            description={description}
            isWorkflowDefault={isDefault}
            blocks={blocks.filter((item) => item.optional)}
          />
        );
      },
      id: "actions",
      disableSortBy: true,
    },
  ];

  if (status === "loading")
    return (
      <StateLoading
        style={{ flex: 1 }}
        title={<Trans id="Fetching workflows" />}
      />
    );

  if (status === "error")
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Could not fetch workflows" />}
      />
    );

  const isDataDefined = typeof data === "object";

  if (!isDataDefined || data.length === 0)
    return (
      <EmptyMessage
        title={<Trans id="You haven't linked any workflows yet" />}
        action={<LinkWorkflow projectId={projectId} />}
      />
    );

  return (
    <>
      <Table striped data={workflowDefs} columns={columns} showIndex />
    </>
  );
}

export default React.memo(ProjectWorkflowList);
