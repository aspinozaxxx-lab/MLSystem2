export function getBrowserLanguage() {
  if (!window.navigator) return null;

  const navigator = window.navigator;
  const agentLanguages = navigator.languages;
  const agentLanguage = agentLanguages
    ? agentLanguages[0]
    : navigator.language || navigator.userLanguage;
  return agentLanguage;
}

export const getLanguageFrom = (languages, fallbackLanguage) => (
  targetLanguage,
) => {
  if (typeof targetLanguage !== "string") return fallbackLanguage;
  const [l] = (targetLanguage || "").toLowerCase().split("-");
  if (languages.indexOf(l) === -1) return fallbackLanguage;
  return l;
};
