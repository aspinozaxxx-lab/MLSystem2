import React from "react";

import { Classes, H5 } from "@blueprintjs/core";

const StatsItem = ({ children, title, topLeft, topRight }) => {
  return (
    <div className="tag">
      {topLeft || topRight ? (
        <div
          className="value"
          style={{ display: "flex", justifyContent: "space-between" }}
        >
          <div>{topLeft}</div>
          <div>{topRight}</div>
        </div>
      ) : (
        <div className="value">
          <div>{children}</div>
        </div>
      )}
      <H5 className={Classes.TEXT_MUTED}>{title}</H5>
    </div>
  );
};

export default StatsItem;
