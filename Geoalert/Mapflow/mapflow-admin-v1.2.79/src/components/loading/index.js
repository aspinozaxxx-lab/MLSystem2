import React from "react";
import { Spinner } from "@blueprintjs/core";

function Loading({ className }) {
  return (
    <div className={className}>
      <Spinner />
    </div>
  );
}

export default React.memo(Loading);
