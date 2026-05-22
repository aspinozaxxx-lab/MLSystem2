import React from "react";
import { Button, Intent } from "@blueprintjs/core";
import { Trans } from "@lingui/react";
import { IconNames } from "@blueprintjs/icons";

import { useGoTo } from "hooks/use-go-to";
import { MAIN } from "constants/routes";
import { EmptyMessage } from "components";

function NotFound() {
  const goToProjects = useGoTo(MAIN);
  return (
    <div className="not-found">
      <EmptyMessage
        iconName={IconNames.ERROR}
        title={<Trans>Not Found</Trans>}
        description={<Trans>Requested resource is not found</Trans>}
        action={
          <Button
            large
            intent={Intent.PRIMARY}
            text={<Trans>Go to Projects</Trans>}
            onClick={goToProjects}
          />
        }
      />
    </div>
  );
}

export default React.memo(NotFound);
