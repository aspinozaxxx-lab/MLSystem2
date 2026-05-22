import React, { useMemo } from "react";
import { gql, useApolloClient } from "@apollo/client";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useAppToaster } from "hooks/use-app-toaster";
import { useParams } from "react-router-dom";
import { t, Trans } from "@lingui/macro";
import { H2, Button, Intent, H5, Tag, InputGroup } from "@blueprintjs/core";

import { Breadcrumbs } from "containers";

import { setTestId } from "test-utils/set-testid";
import { getErrorToast, getSuccessToast } from "toaster";
import Table from "components/table";
import ConfirmDialog from "components/confirm-dialog";
import { IconNames } from "@blueprintjs/icons";
import { GET_WORKFLOWS } from "pages/workflow-list";
import useDebounce from "hooks/use-debounce";
import stable from "stable";
import { sortByDate } from "utils/sort-by-date";
import StateLoading from "components/state-loading";
import EmptyMessage from "components/empty-message";
import { useGoTo } from "hooks/use-go-to";
import * as routes from "constants/routes";
import { useLinkedProjectWorkflowsQuery } from "./queries";
import useProjectQuery from "components/project-workflow-list/queries";

export const LINK_WORKFLOW = gql`
  mutation linkWorkflowDefToProject($workflowDefId: ID!, $projectId: ID!) {
    linkWorkflowDefToProject(
      workflowDefId: $workflowDefId
      projectId: $projectId
    )
  }
`;

export const GET_PROJECT = gql`
  query getProjectById($projectId: ID!) {
    project(id: $projectId) {
      name
      description
      isDefault
      userId
    }
  }
`;

export const GET_USERS = gql`
  query getUsersByEmail($emails: [String!]) {
    users(emails: $emails) {
      id
    }
  }
`;

function LinkWorkflow() {
  const showToast = useAppToaster();
  const goToWorkflowsPage = useGoTo(routes.WORKFLOWS);

  const { projectId } = useParams();

  const client = useApolloClient();

  const [search, setSearch] = React.useState("");
  const debouncedSearch = useDebounce(search, 500);

  const { data: projectData, status: userStatus } = useProjectQuery(projectId);
  const userId = projectData?.userId;

  const {
    data: linkedWDs,
    status: linkedWorkflowsStatus,
    refetch: refetchLinked,
  } = useLinkedProjectWorkflowsQuery(projectId);

  const { data, refetch: refetchAvailable } = useQuery({
    queryKey: ["userAvailabelWorkflows", linkedWDs],
    queryFn: async () => {
      const result = await client.query({
        query: GET_WORKFLOWS,
        fetchPolicy: "no-cache",
        variables: { userIds: [userId] },
      });

      const availableWDs = result?.data?.workflowDefs || [];

      const filteredWDs = availableWDs.filter((wd) => {
        return !linkedWDs.find((linkedWD) => linkedWD.id === wd.id);
      });
      return filteredWDs;
    },
    onError: (e) => {
      showToast(getErrorToast(t`Error fetching available workflows`));
    },
    enabled: userStatus === "success" && linkedWorkflowsStatus === "success",
    cacheTime: 0,
  });

  const availableWorkflows = useMemo(() => {
    if (!data) return undefined;

    let wds = data;
    wds.sort((a, b) => {
      if (a.isDefault === b.isDefault) {
        return 0;
      }
      if (a.isDefault) {
        return -1;
      }
      return 1;
    });
    return stable(wds, sortByDate).filter(({ name }) =>
      name.toLowerCase().includes(debouncedSearch.toLowerCase()),
    );
  }, [data, debouncedSearch]);

  const mutation = useMutation({
    mutationKey: ["linkWorkflowDefToProject", projectId],
    mutationFn: async ({ workflowDefId, projectId }) => {
      const result = await client.mutate({
        mutation: LINK_WORKFLOW,
        variables: { workflowDefId, projectId },
      });
      return result?.data?.linkWorkflowDefToProject;
    },
    onSuccess: async (data) => {
      await refetchLinked();
      await refetchAvailable();
      showToast(getSuccessToast(t`Workflow linked to project`));
    },
    onError: (e) => {
      showToast(getErrorToast(t`Error linking workflow to project`));
    },
  });

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
        Header: <Trans id="Actions" />,
        Cell: ({ row }) => {
          const { id } = row.original;
          return (
            <ConfirmDialog
              className="link-workflow"
              intent={Intent.PRIMARY}
              icon={IconNames.LINK}
              confirmButtonText={<Trans id="Link" />}
              cancelButtonText={<Trans id="Cancel" />}
              text={
                <H5>
                  <Trans id={`Confirm link workflow to project`} />
                </H5>
              }
              onConfirm={(close) => {
                mutation.mutate({
                  workflowDefId: id,
                  projectId,
                });
                close();
              }}
            >
              {({ showDialog }) => (
                <Button
                  minimal
                  elementRef={setTestId`link-workflow`}
                  icon={IconNames.LINK}
                  intent={Intent.PRIMARY}
                  text={<Trans id="Link" />}
                  disabled={mutation.isLoading}
                  onClick={showDialog}
                />
              )}
            </ConfirmDialog>
          );
        },
        id: "actions",
        disableSortBy: true,
      },
    ],
    [mutation, projectId],
  );

  const isLoading =
    userStatus === "loading" ||
    linkedWorkflowsStatus === "loading" ||
    // when `undefined`, means wds are fetching
    !availableWorkflows;

  return (
    <div className="create-edit-workflow">
      <Breadcrumbs />

      <div className="create-edit-workflow__container">
        <H2>
          <Trans>Link Workflows</Trans>
        </H2>
        <div className="manage-workflow-users__search">
          <InputGroup
            className="manage-workflow-users__search-input"
            placeholder={t`Search for workflows by name`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            leftIcon={IconNames.SEARCH}
            rightElement={
              <Button
                minimal
                icon={IconNames.CROSS}
                onClick={() => {
                  setSearch("");
                }}
              />
            }
          />
        </div>

        {isLoading ? (
          <StateLoading
            style={{ flex: 1 }}
            title={<Trans id="Fetching available workflows" />}
          />
        ) : availableWorkflows.length === 0 ? (
          <EmptyMessage
            title={<Trans id="No available workflows to link" />}
            description={
              <Trans id="Either available workflows are already linked or none available" />
            }
            action={
              <Button
                large
                elementRef={setTestId`create-new-project`}
                icon={IconNames.LINK}
                intent={Intent.PRIMARY}
                text={<Trans id="Link workflow to user first" />}
                onClick={goToWorkflowsPage}
              />
            }
          />
        ) : (
          <Table
            striped
            data={availableWorkflows}
            columns={columns}
            showIndex
          />
        )}
      </div>
    </div>
  );
}

export default React.memo(LinkWorkflow);
