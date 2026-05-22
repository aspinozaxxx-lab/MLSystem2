import React from "react";
import { Trans, t } from "@lingui/macro";
import { Button, Intent, H5, Text, ProgressBar } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import ConfirmDialog from "components/confirm-dialog";
import { useGoTo } from "hooks/use-go-to";
import * as routes from "constants/routes";

import { useApolloClient } from "@apollo/client";
import { getErrorToast, showToast } from "toaster";
import { ErrorCodes } from "constants/common";
import useDataProviderUsers from "hooks/use-data-provider-users";
import { useMutation } from "@tanstack/react-query";
import { DELETE_DATA_PROVIDER } from "../queries";

function DataProviderUsersCount({ dataProviderId, goToDataProviderUsers }) {
  const { users, usersLoading } = useDataProviderUsers(dataProviderId);
  
  if (!users || users.length === 0) {
    return (
      <Text>
        <Trans id={`No linked users`} />
      </Text>
    );
  }

  return (
    <div onClick={goToDataProviderUsers} className="unlink-workflow__users">
      <Text className="unlink-workflow__linked">
        <Trans id={`Linked users: `} />
      </Text>

      {usersLoading ? (
        <ProgressBar intent={Intent.PRIMARY} />
      ) : (
        <div className="unlink-workflow__user-emails">
          {users.map((user) => (
            <Text key={user.email} className="">
              {user.email}
            </Text>
          ))}
        </div>
      )}
    </div>
  )
}

function DeleteDataProviderwDialog({ dataProviderId, name, ...btnProps }) {
  const client = useApolloClient();

  const goToDataProviderUsers = useGoTo(routes.DATA_PROVIDER_USERS, {
    dataProviderId,
  });

  const goToDataProviders = useGoTo(routes.DATA_PROVIDERS);

  const deleteDataProviderMutation = useMutation({
    mutationKey: ["deleteDataProvider", dataProviderId],
    mutationFn: async ({ dataProviderId }) => {
      const result = await client.mutate({
        mutation: DELETE_DATA_PROVIDER,
        variables: { id: dataProviderId },
      });
      return result?.data;
    },
    onSuccess: () => {
      showToast({
        message: t`Data provider "${name}" deleted`,
        icon: IconNames.TRASH,
        intent: Intent.WARNING,
      });
    },
    onError: (error) => {
      const { graphQLErrors } = error;
      if (graphQLErrors)
        if (graphQLErrors.some(({ code }) => code === ErrorCodes.WD_IN_USE)) {
          showToast(
            getErrorToast(
              t`Data provider cannot be deleted while it is in use`,
            ),
          );
          return;
        }
      showToast(getErrorToast(t`Error delete data provider "${name}"`));
    },
  });

  return (
    <ConfirmDialog
      className="unlink-workflow"
      intent={Intent.DANGER}
      icon={IconNames.TRASH}
      confirmButtonText={<Trans id="Delete" />}
      cancelButtonText={<Trans id="Cancel" />}
      text={
        <div>
          <H5>
             <Trans>{t`Confirm delete data provider "${name}"`}</Trans>
          </H5>

          <DataProviderUsersCount
            dataProviderId={dataProviderId}
            goToDataProviderUsers={goToDataProviderUsers}
          />
        </div>
      }
      onConfirm={(close) => {
        deleteDataProviderMutation.mutate({ dataProviderId });
        if (dataProviderId) {
          goToDataProviders();
        }
        close();
      }}
    >
      {({ showDialog }) => (
        <Button
          icon={IconNames.TRASH}
          intent={Intent.DANGER}
          text={<Trans id="Delete" />}
          disabled={deleteDataProviderMutation.isLoading}
          onClick={showDialog}
          {...btnProps}
        />
      )}
    </ConfirmDialog>
  );
}

export default React.memo(DeleteDataProviderwDialog);
