import { useMemo, useRef } from "react";
import { Intent } from "@blueprintjs/core";

export function useInputParams(errors, errorMessages = {}) {
  const { required = "This field is required" } = errorMessages;
  const isChangedRef = useRef(false);
  return useMemo(() => {
    let intent = Intent.NONE;
    let helper = "";
    if (errors) {
      if (!isChangedRef.current) isChangedRef.current = true;
      helper = { ...errorMessages, required }[errors.type];
      intent = Intent.DANGER;
    }
    return { intent, helper };
  }, [errors, errorMessages, required]);
}
