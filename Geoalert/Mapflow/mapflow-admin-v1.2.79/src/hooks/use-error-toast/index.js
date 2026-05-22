import { pickToastProps, showToastT } from "toaster";

import { ToastTemplates } from "constants/toasts";
import { useRef, useCallback } from "react";

export function useErrorToast({ message, ...props } = {}) {
  const argsRef = useRef({ message, props });
  return useCallback(
    () =>
      showToastT({
        message: argsRef.current.message,
        ...pickToastProps(argsRef.current.props),
        ...ToastTemplates.ERROR,
      }),
    [],
  );
}
