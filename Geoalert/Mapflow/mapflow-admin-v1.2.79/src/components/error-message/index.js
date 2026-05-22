import React from "react";
import { NonIdealState, Icon, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

function ErrorMessage({
  title,
  description,
  action,
  iconName = IconNames.WARNING_SIGN,
  intent = Intent.DANGER,
  iconSize = 60,
}) {
  const icon = <Icon icon={iconName} intent={intent} iconSize={iconSize} />;
  return (
    <NonIdealState
      icon={icon}
      title={title}
      description={<div>{description}</div>}
      action={action}
    />
  );
}

export default React.memo(ErrorMessage);
