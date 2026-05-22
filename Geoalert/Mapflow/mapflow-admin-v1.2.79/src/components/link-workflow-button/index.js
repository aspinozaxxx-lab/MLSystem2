import React from "react";
import { Trans } from "@lingui/macro";
import { Button, Intent } from "@blueprintjs/core";

import { setTestId } from "test-utils/set-testid";
import * as routes from "constants/routes";
import { IconNames } from "@blueprintjs/icons";
import { generatePath, Link } from 'react-router-dom';

function LinkWorkflowButton({ projectId, ...props }) {
  return (
  <Link to={generatePath(routes.PROJECT_WORKFLOW_LINK, { projectId }, { projectId })}>
    <Button
      outlined
      icon={IconNames.LINK}
      intent={Intent.PRIMARY}
      elementRef={setTestId`link-workflows`}
      text={<Trans id="Link workflows" />}
      {...props}
      large
    />
  </Link>
  );
}

export default React.memo(LinkWorkflowButton);
