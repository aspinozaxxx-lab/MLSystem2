import { Toaster, Position, Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import complement from "ramda/src/complement";
import pickBy from "ramda/src/pickBy";
import isNil from "ramda/src/isNil";
import either from "ramda/src/either";
import includes from "ramda/src/includes";
import __ from "ramda/src/__";

const includesTemplateProps = includes(__, ["icon", "intent"]);
const includesToastProps = includes(__, [
  "icon",
  "intent",
  "action",
  "onDismiss",
  "timeout",
]);
const pickProps = either(complement(isNil), __);

export const pickTemplateProps = pickBy(pickProps(includesTemplateProps));
export const pickToastProps = pickBy(pickProps(includesToastProps));

export const showToastT = ({ message, ...props } = {}, key) => {
  const toasterProps = pickToastProps(props);
  const toast = { message: message, ...toasterProps };
  const toastKey = AppToaster.show(toast, key);
  return {
    key: toastKey,
    dismiss: () => AppToaster.dismiss(toastKey),
  };
};

export const AppToaster = Toaster.create({ position: Position.BOTTOM });

export const showToast = ({ message, ...props } = {}, key) => {
  const toast = { ...props, message: message };
  const toastKey = AppToaster.show(toast, key);
  return {
    key: toastKey,
    dismiss: () => AppToaster.dismiss(toastKey),
  };
};

export const getErrorToast = (message, options = {}) => ({
  message,
  icon: IconNames.ERROR,
  intent: Intent.DANGER,
  ...options,
});

export const getSuccessToast = (message, options = {}) => ({
  message,
  icon: IconNames.TICK,
  intent: Intent.SUCCESS,
  ...options,
});
