import React from "react";
import { AppToaster } from "toaster";

import { ProgressBar } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { useMutation } from "@apollo/client";
import { useGqlError } from "hooks/use-gql-error";

export const getPendingToast = ({ icon, intent }) => {
  return {
    icon,
    timeout: 0,
    message: (
      <ProgressBar
        stripes
        className="pending-toast"
        intent={intent}
        value={1}
      />
    ),
  };
};

export const getSuccessToast = (getMessage, result) => (intent) => {
  return {
    timeout: 0,
    message: getMessage(result),
    icon: IconNames.TICK,
    intent,
  };
};

export const getErrorToast = (getMessage, result) => (intent) => {
  return {
    timeout: 0,
    message: getMessage(result),
    icon: IconNames.WARNING_SIGN,
    intent,
  };
};

export function useMutationWithToasts(
  mutation,
  {
    options,
    errorCodes = {},
    getDefaultErrorMessage,
    getSuccesMessage,
    pendingIntent,
    pendingIcon,
    timeout = 5000,
  },
  updateProcessingCommand,
) {
  const [mutate, result] = useMutation(mutation, options);

  let showKey, updateKey;

  const showResult = (getToast, getMessage) => (result) => {
    const toast = getToast(getMessage, result)(pendingIntent);
    updateKey = AppToaster.show(toast);
  };

  const getErrorMessage = useGqlError(errorCodes, getDefaultErrorMessage);
  const showError = showResult(getErrorToast, getErrorMessage);
  const showSuccess = showResult(getSuccessToast, getSuccesMessage);

  const run = () => {
    const toast = getPendingToast({ icon: pendingIcon, intent: pendingIntent });
    showKey = AppToaster.show(toast);
    mutate()
      .then(showSuccess)
      .catch(showError)
      .finally(() => {
        if (updateProcessingCommand) updateProcessingCommand();
        AppToaster.dismiss(showKey);
        setTimeout(() => AppToaster.dismiss(updateKey), timeout);
      });
  };

  return [run, result];
}
