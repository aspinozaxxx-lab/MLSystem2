import { IconNames } from "@blueprintjs/icons";
import { Intent } from "@blueprintjs/core";

export const ToastTemplates = {
  ERROR: {
    icon: IconNames.WARNING_SIGN,
    intent: Intent.DANGER,
  },
  SUCCESS: {
    icon: IconNames.TICK,
    intent: Intent.SUCCESS,
  },
  WARNING: {
    icon: IconNames.TICK,
    intent: Intent.WARNING,
  },
  INFO: {
    icon: IconNames.INFO_SIGN,
    intent: Intent.PRIMARY,
  },
};
