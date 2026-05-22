import React from "react";
import classnames from "classnames";
import { Classes } from "@blueprintjs/core";

function Subtitle({
  children,
  muted = true,
  fontWeight,
  marginLeft,
  marginTop,
  className,
  fontSize,
  height,
}) {
  return (
    <div
      className={classnames(className, { [Classes.TEXT_MUTED]: muted })}
      style={{
        height: height ? `${height}px` : "unset",
        marginLeft: marginLeft ? `${marginLeft}px` : 0,
        marginTop: marginTop ? `${marginTop}px` : 0,
        fontWeight: fontWeight ? fontWeight : "normal",
        fontSize: fontSize ? fontSize : "unset",
      }}
    >
      {children}
    </div>
  );
}

export default React.memo(Subtitle);
