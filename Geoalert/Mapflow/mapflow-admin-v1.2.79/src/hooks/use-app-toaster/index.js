import { useCallback, useRef } from "react";
import { showToastT, pickToastProps, pickTemplateProps } from "toaster";

import isEmpty from "ramda/src/isEmpty";

export function useAppToaster(defaultTemplate = {}) {
  const defaultTemplateRef = useRef(defaultTemplate);
  const showToast = useCallback(({ message, template = {}, ...props } = {}) => {
    const finalTemplate = isEmpty(template)
      ? defaultTemplateRef.current
      : template;
    return showToastT({
      message,
      ...pickTemplateProps(finalTemplate),
      ...pickToastProps(props),
    });
  }, []);
  return showToast;
}
