import React, { useState, useEffect, useCallback } from "react";
import { setupI18n } from "@lingui/core";
import { I18nProvider } from "@lingui/react";
import { useQuery } from "@apollo/client";

import { GET_LANGUAGE } from "hooks/use-language";
import { LanguageNames } from "constants/common";

export const i18n = setupI18n({
  language: LanguageNames.EN,
  catalogs: { en: {} },
});

export const I18nLoader = ({
  children,
  initialLanguage = LanguageNames.EN,
}) => {
  const [catalogs, setCatalogs] = useState({ [initialLanguage]: {} });

  const { data = {} } = useQuery(GET_LANGUAGE);
  const language = data.language || initialLanguage;

  const [currentLanguage, setCurrentLanguage] = useState(language);

  const loadCatalog = useCallback(async (language) => {
    try {
      setCatalogs((catalogs) => ({ ...catalogs, [language]: {} }));
      setCurrentLanguage(language);
    } catch (error) {
      console.error(`Unnable to fetch catalog for language: ${language}`);
    }
  }, []);

  useEffect(() => {
    loadCatalog(language);
  }, [loadCatalog, language]);

  return (
    <I18nProvider language={currentLanguage} catalogs={catalogs}>
      {children}
    </I18nProvider>
  );
};

export default React.memo(I18nLoader);
