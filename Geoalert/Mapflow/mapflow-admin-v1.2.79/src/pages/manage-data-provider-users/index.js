import { useApolloClient } from "@apollo/client";
import { Button, H2, H5, InputGroup, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { t, Trans } from "@lingui/macro";
import { useMutation } from "@tanstack/react-query";
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
import { useSearchUser } from "hooks/use-search-users";

import UserSearchCallout from "../../components/user-search-callout";
import DeleteDataProviderDialog from "components/data-provider/delete-data-provider-dialog";
import {
  LINK_DATA_PROVIDER,
  UNLINK_DATA_PROVIDER,
} from "components/data-provider/queries";
import useDataProviderUsers from "hooks/use-data-provider-users";

function ManageDataProviderUsers() {
  const client = useApolloClient();
  const { dataProviderId } = useParams();

  const [searchInput, setSearch] = useState("");
  const debouncedSearchInput = useDebounce(searchInput, 500);

  const { searchedUser, searchedUserLoading } = useSearchUser(
    debouncedSearchInput,
  );

  const { users, usersLoading, refetchUsers } = useDataProviderUsers(
    dataProviderId,
  );

  const linkMutation = useMutation({
    mutationKey: ["linkDataProviderUser", dataProviderId],
    mutationFn: async ({ dataProviderId, userId, email }) => {
      const result = await client.mutate({
        mutation: LINK_DATA_PROVIDER,
        variables: { dataProviderId, userId },
      });
      return result?.data;
    },
    onSuccess: (data) => {
      refetchUsers();
      showToast(getSuccessToast(t`Data provider linked to user`));
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error linking Data provider to user`));
    },
  });

  const unlinkMutation = useMutation({
    mutationKey: ["unlinkDataProviderUser", dataProviderId],
    mutationFn: async ({ dataProviderId, userId, email }) => {
      const result = await client.mutate({
        mutation: UNLINK_DATA_PROVIDER,
        fetchPolicy: "no-cache",
        variables: { dataProviderId, userId },
      });

      return result?.data;
    },
    onSuccess: () => {
      refetchUsers();
      showToast({
        message: t`Data provider unlinked from user`,
        icon: IconNames.TRASH,
        intent: Intent.WARNING,
      });
    },
    onError: (error) => {
      showToast(getErrorToast(t`Error unlink Data provider`));
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
                  <Trans>{t`Confirm unlink data provider from user "${email}"`}</Trans>
                </H5>
              }
              onConfirm={(close) => {
                unlinkMutation.mutate({ dataProviderId, userId: id, email });
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
    [unlinkMutation, dataProviderId, usersLoading],
  );

  return (
    <div className="processings">
      <Breadcrumbs />
      <div className="processings__container">
        <div className="manage-workflow-users__header">
          <H2>
            <Trans>Manage Data Provider Users</Trans>
          </H2>

          {!usersLoading && users.length === 0 && (
            <DeleteDataProviderDialog
              outlined
              large
              dataProviderId={dataProviderId}
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
                <Trans>{t`Confirm link data provider to user "${searchedUser?.email}"`}</Trans>
              </H5>
            }
            onConfirm={(close) => {
              linkMutation.mutate({ dataProviderId, userId: searchedUser?.id });
              close();
            }}
          >
            {({ showDialog }) => (
              <Button
                elementRef={setTestId`link-dataProvider`}
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
            title={<Trans id="Fetching users linked to data provider" />}
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

export default React.memo(ManageDataProviderUsers);
