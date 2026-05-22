import React, { useCallback } from "react";
import { Trans, t } from "@lingui/macro";
import { Button, H1, InputGroup } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

function ProcessingsHeader({ search, setSearch }) {
  const handleClear = useCallback(() => {
    setSearch("");
  }, [setSearch]);

  return (
    <>
      <div className="processings-header">
        <div className="processings-header__title">
          <H1>
            <Trans>Processings</Trans>
          </H1>
        </div>
        <div className="processings-header__search-input">
          <InputGroup
            placeholder={t`Enter to search for processings by email, wd name or processing name`}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            leftIcon={IconNames.SEARCH}
            rightElement={
              <Button minimal icon={IconNames.CROSS} onClick={handleClear} />
            }
          />
        </div>
      </div>
    </>
  );
}

export default React.memo(ProcessingsHeader);
