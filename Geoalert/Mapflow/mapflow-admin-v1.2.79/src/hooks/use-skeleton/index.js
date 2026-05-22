import { useMemo } from "react";
import classnames from "classnames";
import { Classes } from "@blueprintjs/core";

const skeletonWrapper = (loading = true) => (...classes) =>
  classnames(classes, { [Classes.SKELETON]: loading });

function useSkeleton(loading = true) {
  const skeletoned = useMemo(() => skeletonWrapper(loading), [loading]);
  return { skeletoned };
}

export { useSkeleton };
