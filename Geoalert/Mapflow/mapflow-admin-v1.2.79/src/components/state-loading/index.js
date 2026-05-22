import React from "react";
import { Spinner, NonIdealState } from "@blueprintjs/core";

function StateLoading({ className, title, description }) {
  return (
    <NonIdealState
      className={className}
      icon={<Spinner />}
      title={title}
      description={<div>{description}</div>}
    />
  );
}

export default React.memo(StateLoading);
