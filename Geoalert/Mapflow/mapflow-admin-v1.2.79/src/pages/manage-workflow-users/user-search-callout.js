import { Callout, Intent, Text } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { Trans } from "@lingui/macro";
import React from "react";

function UserSearchCallout({ searchedUser, searchedUserLoading }) {
  if (searchedUserLoading)
    return (
      <Callout
        className="manage-workflow-users__search-result"
        intent={Intent.NONE}
        icon={IconNames.SEARCH_TEXT}
      >
        <Text>
          <Trans>Search in progress...</Trans>
        </Text>
      </Callout>
    );

  return (
    <Callout
      className="manage-workflow-users__search-result"
      intent={!searchedUser ? Intent.DANGER : Intent.SUCCESS}
      icon={!searchedUser ? IconNames.CANCEL : IconNames.SUCCESS}
    >
      {!searchedUser ? (
        <Text>
          <Trans id={`User with this email doesn't exist`} />
        </Text>
      ) : (
        <Text>
          <Trans id={`Found user with this email`} />
        </Text>
      )}
    </Callout>
  );
}

export default React.memo(UserSearchCallout);
