import { DEFAULT_DATE_FORMAT } from "constants/common";
import { useLingui } from "@lingui/react";

const ToLocaleTime = ({ time }) => {
  const { i18n } = useLingui();

  return i18n.date(time, DEFAULT_DATE_FORMAT);
};

export default ToLocaleTime;
