import { gql, useApolloClient } from "@apollo/client";
import { Button, H2, H5, InputGroup, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { t, Trans } from "@lingui/macro";
import { useMutation, useQuery } from "@tanstack/react-query";
import ConfirmDialog from "components/confirm-dialog";
import EmptyMessage from "components/empty-message";
import StateLoading from "components/state-loading";
import Table from "components/table";
import Breadcrumbs from "containers/breadcrumbs";
import useDebounce from "hooks/use-debounce";
import React, { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { setTestId } from "test-utils/set-testid";
import { getErrorToast, getSuccessToast, showToast } from "toaster";
import DeleteWorkflowDialog from "pages/workflow-list/delete-workflow-dialog";
import { useSearchUser } from "hooks/use-search-users";

import UserSearchCallout from '../../components/user-search-callout'

export const GET_USERS_LINKED_TO_WORKFLOW = gql`
  query getWorkflowDefUsers($id: ID!) {
    workflowDefUsers(id: $id) {
      id
      email
    }
  }
`;

export const LINK_WORKFLOW_TO_USER = gql`
  mutation linkWorkflowToUser($userId: ID!, $workflowDefId: ID!) {
    linkWorkflowDefToUser(userId: $userId, workflowDefId: $workflowDefId)
  }
`;

const UNLINK_WORKFLOW_FROM_USER = gql`
  mutation unlinkWorkflowFromUser($userId: ID!, $workflowDefId: ID!) {
    unlinkWorkflowDefFromUser(userId: $userId, workflowDefId: $workflowDefId)
  }
`;

function ManageWorkflowUsers() {
  const client = useApolloClient();
  const { workflowDefId } = useParams();

  const [searchInput, setSearch] = useState("");
  const debouncedSearchInput = useDebounce(searchInput, 500);

  const { searchedUser, searchedUserLoading } = useSearchUser(debouncedSearchInput)


  const {
    data: users,
    isLoading: usersLoading,
    refetch: refetchUsers,
  } = useQuery({
    queryKey: ["workflowDefUsers", workflowDefId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_USERS_LINKED_TO_WORKFLOW,
        fetchPolicy: "no-cache",
        variables: { id: workflowDefId },
      });
      return result?.data?.workflowDefUsers;
    },
    onError: () => {
      showToast(getErrorToast(t`Failed to get users linked to workflow`));
    },
  });

  const linkMutation = useMutation({
    mutationKey: ["linkWorkflowDefToUser", workflowDefId],
    mutationFn: async ({ workflowDefId, userId, email }) => {
      const result = await client.mutate({
        mutation: LINK_WORKFLOW_TO_USER,
        variables: { workflowDefId, userId },
      });
      return result?.data?.linkWorkflowDefToUser;
    },
    onSuccess: (data) => {
      refetchUsers();
      showToast(getSuccessToast(t`Workflow linked to user`));
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error linking Workflow to user`));
    },
  });

  const unlinkMutation = useMutation({
    mutationKey: ["unlinkWorkflowDefFromUser", workflowDefId],
    mutationFn: async ({ workflowDefId, userId, email }) => {
      const result = await client.mutate({
        mutation: UNLINK_WORKFLOW_FROM_USER,
        fetchPolicy: "no-cache",
        variables: { workflowDefId, userId },
      });
      return result?.data?.unlinkWorkflowDefFromUser;
    },
    onSuccess: () => {
      refetchUsers();
      showToast({
        message: t`Workflow unlinked from user`,
        icon: IconNames.TRASH,
        intent: Intent.WARNING,
      });
    },
    onError: (error) => {
      showToast(getErrorToast(t`Error unlink workflow`));
    },
  });

  const columns = useMemo(
    () => [
      {
        Header: <Trans id="Email" />,
        id: "email",
        accessor: "email",
      },
      {
        Header: <Trans id="Actions" />,
        id: "linked",
        Cell: ({ row }) => {
          const { id, email } = row.original;

          return (
            <ConfirmDialog
              className="unlink-workflow"
              intent={Intent.DANGER}
              icon={IconNames.UNDO}
              confirmButtonText={<Trans id="Unlink" />}
              cancelButtonText={<Trans id="Cancel" />}
              text={
                <H5>
                  <Trans>{t`Confirm unlink workflow from user "${email}"`}</Trans>
                </H5>
              }
              onConfirm={(close) => {
                unlinkMutation.mutate({ workflowDefId, userId: id, email });
                close();
              }}
            >
              {({ showDialog }) => (
                <Button
                  minimal
                  elementRef={setTestId`unlink-workflow`}
                  icon={IconNames.UNDO}
                  intent={Intent.DANGER}
                  text={<Trans id="Unlink" />}
                  disabled={usersLoading}
                  onClick={showDialog}
                />
              )}
            </ConfirmDialog>
          );
        },
      },
    ],
    [unlinkMutation, usersLoading, workflowDefId],
  );

  return (
    <div className="processings">
      <Breadcrumbs />
      <div className="processings__container">
        <div className="manage-workflow-users__header">
          <H2>
            <Trans>Manage Workflow Users</Trans>
          </H2>

          {!usersLoading && users.length === 0 && (
            <DeleteWorkflowDialog
              outlined
              large
              workflowDefId={workflowDefId}
            />
          )}
        </div>

        <div className="manage-workflow-users__search">
          <InputGroup
            className="manage-workflow-users__search-input"
            placeholder={t`Search and link users by email`}
            value={searchInput}
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

          <ConfirmDialog
            className="unlink-workflow"
            intent={Intent.PRIMARY}
            icon={IconNames.LINK}
            confirmButtonText={<Trans id="Link" />}
            cancelButtonText={<Trans id="Cancel" />}
            text={
              <H5>
                <Trans>{t`Confirm link workflow to user "${searchedUser?.email}"`}</Trans>
              </H5>
            }
            onConfirm={(close) => {
              linkMutation.mutate({ workflowDefId, userId: searchedUser?.id });
              close();
            }}
          >
            {({ showDialog }) => (
              <Button
                elementRef={setTestId`link-workflow`}
                icon={IconNames.LINK}
                intent={Intent.PRIMARY}
                text={<Trans id="Link" />}
                disabled={
                  searchedUserLoading || !searchedUser || !debouncedSearchInput
                }
                onClick={showDialog}
              />
            )}
          </ConfirmDialog>
        </div>

        {debouncedSearchInput && (
          <UserSearchCallout
            searchedUser={searchedUser}
            searchedUserLoading={searchedUserLoading}
          />
        )}

        {usersLoading ? (
          <StateLoading
            style={{ flex: 1 }}
            title={<Trans id="Fetching users linked to workflow" />}
          />
        ) : users.length === 0 ? (
          <EmptyMessage
            iconName={IconNames.PEOPLE}
            title={<Trans id="You haven't linked any users yet" />}
            description={
              <Trans>
                To link users, enter emails above and click "Link" button.
              </Trans>
            }
          />
        ) : (
          <Table striped data={users} columns={columns} />
        )}
      </div>
    </div>
  );
}

export default React.memo(ManageWorkflowUsers);
