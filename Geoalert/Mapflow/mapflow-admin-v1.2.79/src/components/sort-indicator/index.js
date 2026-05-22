import React from "react";
import classnames from "classnames";
import { Icon } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

function SortIndicator({ className, isSorted, sortDirection }) {
  const iconName = isSorted
    ? sortDirection === "DESC"
      ? IconNames.CARET_DOWN
      : IconNames.CARET_UP
    : IconNames.DOUBLE_CARET_VERTICAL;
  const iconClasses = classnames("sort-indicator", {
    "sort-indicator--selected": isSorted,
  });
  return (
    <span className={classnames(iconClasses, className)}>
      <Icon icon={iconName} />
    </span>
  );
}

export default React.memo(SortIndicator);
