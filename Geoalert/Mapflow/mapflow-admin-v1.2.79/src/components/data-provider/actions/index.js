import React from "react";
import { Trans } from "@lingui/macro";
import { Button, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

import { setTestId } from "test-utils/set-testid";
import * as routes from "constants/routes";
import { useGoTo } from "hooks/use-go-to";
import { Link, generatePath } from "react-router-dom";

import DeleteDataProviderwDialog from '../delete-data-provider-dialog'

function DataProviderActions({ dataProviderId, isDefault, name }) {
  const goToDataProviderUsers = useGoTo(routes.DATA_PROVIDER_USERS, {
    dataProviderId,
  });
  const goToDataProviderEdit = useGoTo(routes.DATA_PROVIDER_EDIT, {
    dataProviderId,
  });

  return (
    <div>
      <Link to={generatePath(routes.DATA_PROVIDER_EDIT, { dataProviderId })}>
        <Button
          minimal
          elementRef={setTestId`edit-dataProvider`}
          icon={IconNames.EDIT}
          intent={Intent.SUCCESS}
          text={<Trans id="Edit" />}
          onClick={goToDataProviderEdit}
        />
      </Link>

      {!isDefault && (
        <Link to={generatePath(routes.DATA_PROVIDER_USERS, { dataProviderId })}>
          <Button
            minimal
            elementRef={setTestId`manage-dataProvider-users`}
            icon={IconNames.USER}
            intent={Intent.PRIMARY}
            text={<Trans id="Manage users" />}
            onClick={goToDataProviderUsers}
          />
        </Link>
      )}

      <DeleteDataProviderwDialog dataProviderId={dataProviderId} name={name} minimal />
    </div>
  );
}

export default React.memo(DataProviderActions);
