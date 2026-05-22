import { format, parse } from "date-fns";
import * as locales from "date-fns/locale";

function maybeGetLocaleOptions(localeCode) {
  if (locales[localeCode] !== undefined) {
    return { locale: locales[localeCode] };
  }
  return undefined;
}

export function getDateFnsFormatter(formatStr) {
  return {
    formatDate: (date, localeCode) =>
      format(date, formatStr, maybeGetLocaleOptions(localeCode)),
    parseDate: (str, localeCode) =>
      parse(str, formatStr, new Date(), maybeGetLocaleOptions(localeCode)),
    placeholder: `${formatStr}`,
  };
}

export const DATE_FNS_FORMATS = {
  MMddyyyy: getDateFnsFormatter("MM/dd/yyyy"),
  yyyyMMdd: getDateFnsFormatter("yyyy-MM-dd"),
  yyyyMMddHHmmSS: getDateFnsFormatter("yyyy-MM-dd HH:mm:ss"),
  ddMMyyyy: getDateFnsFormatter("dd/MM/yyyy"),
};
