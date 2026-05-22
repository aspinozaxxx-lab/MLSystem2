/* istanbul ignore file */
import { ApolloClient } from "@apollo/client";
import { InMemoryCache } from "@apollo/client/cache";

import { resolvers, typeDefs } from "./resolvers";
import { cacheRedirects } from "./cache-redirects";
import appLink from "./link";
import { GET_LANGUAGE } from "hooks/use-language";
import { GET_THEME } from "hooks/use-theme";

import { ThemeNames, LanguageNames } from "constants/common";

import {
  getLanguageFrom,
  getBrowserLanguage,
} from "utils/get-browser-language";

const cache = new InMemoryCache({ cacheRedirects });

const client = new ApolloClient({
  credentials: "include",
  cache,
  link: appLink,
  resolvers: [...resolvers],
  typeDefs: [...typeDefs],
});

const browserLanguage = getBrowserLanguage();
const savedLanguage = localStorage.getItem("language");
const getUserLanguage = getLanguageFrom(
  [LanguageNames.RU, LanguageNames.EN],
  LanguageNames.EN,
);
const language = getUserLanguage(
  savedLanguage ? savedLanguage : browserLanguage,
);

const theme = localStorage.getItem("theme") || ThemeNames.LIGHT;

cache.writeQuery({
  query: GET_LANGUAGE,
  data: {
    language: language,
  },
});

cache.writeQuery({
  query: GET_THEME,
  data: {
    theme: theme,
  },
});

export default client;
