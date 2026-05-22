import React, { useMemo } from "react";
import { useHistory } from "react-router-dom";
import * as routes from "constants/routes";
import useDebounce from "hooks/use-debounce";

import { Trans } from "@lingui/macro";
import { useQuery } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";
import stable from "stable";

import { POLL_INTERVAL } from "constants/envs";

import Actions from "./actions";
import WorkflowsHeader from "./workflows-header";
import { sortByDate } from "utils/sort-by-date";
import { Tag, Text } from "@blueprintjs/core";
import WorkflowsListContent from "./workflows-list-content";
import { useWorkflowSearch } from "app";
import ToLocaleTime from "components/toLocalTime";

export const GET_WORKFLOWS = gql`
  query listWorkflows(
    $ids: [ID!]
    $userIds: [ID!]
    $projectIds: [ID!]
    $isDefault: Boolean
  ) {
    workflowDefs(
      ids: $ids
      userIds: $userIds
      projectIds: $projectIds
      isDefault: $isDefault
    ) {
      id
      name
      description
      created
      updated
      isDefault
    }
  }
`;

function WorkflowList() {
  const { search } = useWorkflowSearch();
  const debouncedSearch = useDebounce(search, 500);
  const history = useHistory();
  const client = useApolloClient();

  const { data, status } = useQuery({
    queryKey: ["workflowsDefs"],
    queryFn: async () => {
      const result = await client.query({
        query: GET_WORKFLOWS,
        fetchPolicy: "no-cache",
      });
      return result?.data?.workflowDefs;
    },
    refetchInterval: POLL_INTERVAL,
    refetchOnWindowFocus: false,
    keepPreviousData: true,
  });

  const workflowDefs = useMemo(() => {
    let wds = data || [];
    wds = stable(wds, sortByDate);
    wds.sort((a, b) => {
      if (a.isDefault === b.isDefault) {
        return 0;
      }
      if (a.isDefault) {
        return -1;
      }
      return 1;
    });
    return wds.filter(({ name }) =>
      name.toLowerCase().includes(debouncedSearch.toLowerCase()),
    );
  }, [data, debouncedSearch]);

  const columns = useMemo(
    () => [
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
        Header: <Trans id="Created" />,
        id: "created",
        accessor: "created",
        Cell: (cell) => {
          return (
            <Text className="workflow-cell__date">
              {<ToLocaleTime time={cell.value} />}
            </Text>
          );
        },
      },
      {
        Header: <Trans id="Updated" />,
        id: "updated",
        accessor: "updated",
        Cell: (cell) => (
          <Text className="workflow-cell__date">
            {<ToLocaleTime time={cell.value} />}
          </Text>
        ),
      },
      {
        Header: <Trans id="Description" />,
        id: "description",
        accessor: "description",
        Cell: ({ row }) => (
          <Text className="workflow-cell__description" ellipsize>
            {row.original.description}
          </Text>
        ),
      },
      {
        Header: <Trans id="Actions" />,
        Cell: ({ row }) => {
          const { id, name, isDefault } = row.original;
          return (
            <Actions isDefault={isDefault} name={name} workflowDefId={id} />
          );
        },
        id: "actions",
        disableSortBy: false,
      },
    ],
    [],
  );

  return (
    <div className="projects">
      <WorkflowsHeader onCreate={() => history.push(routes.WORKFLOW_CREATE)} />

      <WorkflowsListContent
        status={status}
        workflowDefs={workflowDefs}
        columns={columns}
      />
    </div>
  );
}

export default React.memo(WorkflowList);
