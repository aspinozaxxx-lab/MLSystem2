import React from "react";
import { Icon, Text } from "@blueprintjs/core";

export function TabTitle({ icon, text, textIconLeft, textIconRight }) {
  return (
    <div className="tab-title">
      <Icon className="icon" icon={icon} />
      <div className="text">
        <div className="tab__title">
          {textIconLeft && (
            <Icon className="text-icon__left" icon={textIconLeft} />
          )}

          <Text>{text}</Text>

          {textIconRight && (
            <Icon className="text-icon__right" icon={textIconRight} />
          )}
        </div>
      </div>
    </div>
  );
}
