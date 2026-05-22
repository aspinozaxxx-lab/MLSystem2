import React from "react";
import { Trans, t } from "@lingui/macro";
import { Button, Intent, H1, InputGroup } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { setTestId } from "test-utils/set-testid";
import { Link } from "react-router-dom";
import * as routes from "constants/routes";

function DataProvidersHeader({ onCreate,dataProviderSearch, setDataProviderSearch  }) {

  return (
    <div className="projects-header">
      <div className="projects-header__title">
        <H1>
          <Trans>Data Providers</Trans>
        </H1>

        <div>
          <Link to={routes.DATA_PROVIDER_CREATE}>
            <Button
              large
              minimal
              elementRef={setTestId`create-data-provider`}
              intent={Intent.PRIMARY}
              icon={IconNames.PLUS}
              text={<Trans id="Create data provider" />}
              onClick={onCreate}
            />
          </Link>
        </div>
      </div>

      <InputGroup
        className="projects-header__search-input"
        placeholder={t`Search for data providers by name (case sensitive)`}
        value={dataProviderSearch}
        onChange={(e) => setDataProviderSearch(e.target.value)}
        leftIcon={IconNames.SEARCH}
        rightElement={
          <Button
            minimal
            icon={IconNames.CROSS}
            onClick={() => {
              setDataProviderSearch("");
            }}
          />
        }
      />
    </div>
  );
}

export default React.memo(DataProvidersHeader);
