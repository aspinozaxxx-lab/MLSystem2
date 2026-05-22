import React, { useCallback, useState } from "react";
import { Trans } from "@lingui/macro";
import { Button } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { useApolloClient } from "@apollo/client";

import { setTestId } from "test-utils/set-testid";
import { avanpostCookies } from "providers/auth-avanpost-provider";
import { useAuth } from "providers/auth-avanpost-provider/AuthProvider";
import { endSession } from "shared/api/client/avanpost";

function SignOutButton({ minimal }) {
  const client = useApolloClient();
  const { handleLogout } = useAuth();
  const [isLoading, setIsLoading] = useState(false);

  const signOut = useCallback(async () => {
    // TODO: avanpost auth logout
    try {
      setIsLoading(true);

      await endSession(avanpostCookies.getTokens().token);
      avanpostCookies.clearSession();
      handleLogout();

      setIsLoading(false);
    } catch (error) {
      setIsLoading(false);
      console.error(error);
    }

    // client.writeData({ data: { isLoggedIn: false, userRole: null } });
  }, [handleLogout]);

  return (
    <div>
      <Button
        icon={IconNames.LOG_OUT}
        text={<Trans id="Sign Out" />}
        minimal={minimal}
        elementRef={setTestId`sign-out-button`}
        loading={isLoading}
        disabled={isLoading}
        onClick={() => signOut().then(setTimeout(client.resetStore, 100))}
      />
    </div>
  );
}

export default React.memo(SignOutButton);
