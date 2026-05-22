import React, { useMemo } from "react";
import classnames from "classnames";
import { Text } from "@blueprintjs/core";

import { useSkeleton } from "hooks/use-skeleton";

function TextLoader({
  className = "",
  text = null,
  skip = false,
  loading = null,
  ellipsize = false,
  length = 5,
  fill = false,
  wrapperTagName = "span",
}) {
  const Component = useMemo(() => (fill ? "div" : wrapperTagName), [
    fill,
    wrapperTagName,
  ]);
  const textClassName = useMemo(
    () => classnames(className, { "full-width": fill }),
    [className, fill],
  );
  const isLoading = typeof loading === "boolean" ? loading : text === null;
  const { skeletoned } = useSkeleton(isLoading);

  const repeat = ([r]) => r.repeat(parseInt(length));
  return skip ? null : (
    <Text ellipsize={ellipsize} className={textClassName}>
      <Component className={skeletoned``}>
        {isLoading ? repeat`_` : text}
      </Component>
    </Text>
  );
}

export default React.memo(TextLoader);
