import { useApolloClient } from "@apollo/client";
import { Button } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { Trans } from "@lingui/macro";
import ErrorMessage from "components/error-message";
import { useCallback } from "react";
import { avanpostCookies } from "../model";

export const AccessDenied = (props) => {
  const client = useApolloClient();

  const signOut = useCallback(async () => {
    // TODO Avanpost auth logout
    avanpostCookies.clearSession();
    client.writeData({ data: { isLoggedIn: false, userRole: null } });
  }, [client]);

  return (
    <ErrorMessage
      title={<Trans>Access denied</Trans>}
      description={
        <Trans>Only administrators allowed to use this service</Trans>
      }
      action={
        <Button
          icon={IconNames.LOG_OUT}
          text={<Trans id="Sign Out" />}
          onClick={() => signOut().then(setTimeout(client.resetStore, 100))}
        />
      }
    />
  );
};
