import React from "react";
import { Trans, t } from "@lingui/macro";
import { Button, Intent, H1, InputGroup } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { setTestId } from "test-utils/set-testid";
import { Link } from "react-router-dom";
import * as routes from "constants/routes";
import { useWorkflowSearch } from "app";

function WorkflowsHeader({ onCreate }) {
  const { search, setSearch } = useWorkflowSearch();

  return (
    <div className="projects-header">
      <div className="projects-header__title">
        <H1>
          <Trans>Workflows</Trans>
        </H1>

        <div>
          <Link to={routes.WORKFLOW_CREATE}>
            <Button
              large
              minimal
              elementRef={setTestId`create-workflow`}
              intent={Intent.PRIMARY}
              icon={IconNames.PLUS}
              text={<Trans id="Create workflow" />}
              onClick={onCreate}
            />
          </Link>
        </div>
      </div>

      <InputGroup
        className="projects-header__search-input"
        placeholder={t`Search for workflows by name (case sensitive)`} 
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
  );
}

export default React.memo(WorkflowsHeader);
