import { t } from "@lingui/macro";
import { Intent } from "@blueprintjs/core";

const THEME_LIGHT = "light";
const THEME_DARK = "dark";

const LANGUAGE_RU = "ru";
const LANGUAGE_EN = "en";

export const ThemeNames = {
  DARK: THEME_DARK,
  LIGHT: THEME_LIGHT,
  T: {
    [THEME_LIGHT]: t`light`,
    [THEME_DARK]: t`dark`,
  },
};

export const LanguageNames = {
  EN: LANGUAGE_EN,
  RU: LANGUAGE_RU,
  T: {
    [LANGUAGE_EN]: t`english`,
    [LANGUAGE_RU]: t`russian`,
  },
};

export const STATUS_SUCCESS = "OK";
export const STATUS_FAILED = "FAILED";
export const STATUS_PENDING = "IN_PROGRESS";
export const STATUS_UNPROCESSED = "UNPROCESSED";
export const STATUS_CANCELLED = "CANCELLED";

export const Statuses = {
  SUCCESS: STATUS_SUCCESS,
  FAILED: STATUS_FAILED,
  PENDING: STATUS_PENDING,
  UNPROCESSED: STATUS_UNPROCESSED,
  STATUS_CANCELLED: STATUS_CANCELLED,
};

export const ProgressStatuses = {
  T: {
    [STATUS_SUCCESS]: t`success`,
    [STATUS_FAILED]: t`failed`,
    [STATUS_PENDING]: t`in progress`,
    [STATUS_UNPROCESSED]: t`unprocessed`,
    [STATUS_CANCELLED]: t`cancelled`,
  },
  I: {
    [STATUS_SUCCESS]: Intent.SUCCESS,
    [STATUS_FAILED]: Intent.DANGER,
    [STATUS_PENDING]: Intent.WARNING,
    [STATUS_CANCELLED]: Intent.NONE,
    [STATUS_UNPROCESSED]: Intent.NONE,
  },
};

export const TRANSLATIONS = {
  завершено: STATUS_SUCCESS,
  ошибка: STATUS_FAILED,
  "в работе": STATUS_PENDING,
  "не обработано": STATUS_UNPROCESSED,
  приостановлено: STATUS_CANCELLED,
};

const ROLE_USER = "USER";
const ROLE_ADMIN = "ADMIN";

export const Roles = {
  USER: ROLE_USER,
  ADMIN: ROLE_ADMIN,
};

export const DEFAULT_DATE_FORMAT = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
};

const ERROR_WD_IN_USE = "WD_IN_USE";
export const ErrorCodes = {
  WD_IN_USE: ERROR_WD_IN_USE,
};
