import React from "react";
import { Classes } from "@blueprintjs/core";
import { Trans } from "@lingui/macro";

export const LoadingItem = React.memo(() => (
  <div className="aoi-list-item-skeleton">
    <div className={Classes.SKELETON} />
    <div>
      <Trans>Loading</Trans>...
    </div>
  </div>
));
