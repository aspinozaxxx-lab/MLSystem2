import React, { useEffect, useState } from "react";
import { i18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { useQuery } from "@apollo/client";
import { en, ru } from "make-plural/plurals";

import { GET_LANGUAGE } from "hooks/use-language";
import { LanguageNames } from "constants/common";

// i18n.loadLocaleData(LanguageNames.EN, {plurals: en});
i18n.loadLocaleData(LanguageNames.RU, { plurals: ru });
// i18n.load("en", require("../locales/en/messages").messages);
i18n.load("ru", require("../locales/ru/messages").messages);

export const I18nLoader = ({
  children,
  initialLanguage = LanguageNames.RU,
}) => {
  const language = LanguageNames.RU;

  const [currentLanguage, setCurrentLanguage] = useState(null);

  useEffect(() => {
    if (i18n.locale !== language) {
      console.log("Changing locale from " + i18n.locale + " to " + language);
      i18n.activate(language);
      setCurrentLanguage(language);
    }
  }, [language]);

  if (!currentLanguage) {
    return null;
  }

  return <I18nProvider i18n={i18n}>{children}</I18nProvider>;
};

export default React.memo(I18nLoader);
