import React from "react";
import { NonIdealState, Icon, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

function EmptyMessage({ iconName, title, description, action }) {
  const icon = (
    <Icon
      icon={iconName || IconNames.FOLDER_OPEN}
      intent={Intent.NONE}
      iconSize={60}
    />
  );
  return (
    <NonIdealState
      className="empty-message"
      icon={icon}
      title={title}
      description={<div>{description}</div>}
      action={action}
    />
  );
}

export default React.memo(EmptyMessage);
