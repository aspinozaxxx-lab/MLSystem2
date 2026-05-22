import { gql, useQuery, useMutation } from "@apollo/client";

import { LanguageNames } from "constants/common";

export const GET_LANGUAGE = gql`
  query getCurrentLanguage {
    language @client
  }
`;

export const TOGGLE_LANGUAGE = gql`
  mutation toggleLanguage {
    toggleLanguage @client
  }
`;

export function useLanguage(initialLanguage = LanguageNames.EN) {
  const { data = {} } = useQuery(GET_LANGUAGE);
  const [toggleLanguage] = useMutation(TOGGLE_LANGUAGE);
  const language = data.language || initialLanguage;
  return { language, toggleLanguage };
}
