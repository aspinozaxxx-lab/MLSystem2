import React from "react";
import { Trans, t } from "@lingui/macro";
import { Button, Intent, H1, InputGroup } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { setTestId } from "test-utils/set-testid";
import { useProjectSearch } from "app";

function ProjectsHeader({ onCreate }) {
  const { search, setSearch } = useProjectSearch();

  return (
    <>
      <div className="projects-header">
        <div className="projects-header__title">
          <H1>
            <Trans>Projects</Trans>
          </H1>
          <div>
            <Button
              large
              minimal
              elementRef={setTestId`create-new-project`}
              intent={Intent.PRIMARY}
              icon={IconNames.PLUS}
              text={<Trans id="Create project" />}
              onClick={onCreate}
            />
          </div>
        </div>
        <div className="projects-header__search-input">
          <InputGroup
            placeholder={t`Enter to search for projects by name, description or author`} 
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
      </div>
    </>
  );
}

export default ProjectsHeader;
