import { GET_LANGUAGE } from "hooks/use-language";
import { LanguageNames } from "constants/common";
import { gql } from "@apollo/client";

const typeDefs = gql`
  enum Language {
    en
    ru
  }

  extend type Query {
    language: Language!
  }

  extend type Mutation {
    toggleLanguage: Language
  }
`;

const { EN, RU } = LanguageNames;

const resolvers = {
  Mutation: {
    toggleLanguage: (_, __, { cache }) => {
      const { language } = cache.readQuery({ query: GET_LANGUAGE });

      const nextLanguage = language === EN ? RU : language === RU ? EN : EN;
      localStorage.setItem("language", nextLanguage);

      cache.writeQuery({
        query: GET_LANGUAGE,
        data: {
          language: nextLanguage,
        },
      });
      return null;
    },
  },
};

const language = { typeDefs, resolvers };
export default language;
