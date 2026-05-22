import { gql } from "@apollo/client";

import { GET_THEME } from "hooks/use-theme";
import { ThemeNames } from "constants/common";

const typeDefs = gql`
  enum Theme {
    light
    dark
  }

  extend type Query {
    theme: Theme!
  }

  extend type Mutation {
    toggleTheme: Theme!
  }
`;

const { DARK, LIGHT } = ThemeNames;

const resolvers = {
  Mutation: {
    toggleTheme: (_, __, { cache }) => {
      const { theme } = cache.readQuery({ query: GET_THEME });

      const nextTheme = theme === LIGHT ? DARK : LIGHT;
      localStorage.setItem("theme", nextTheme);

      return cache.writeQuery({
        query: GET_THEME,
        data: {
          theme: nextTheme,
        },
      });
    },
  },
};

const theme = { typeDefs, resolvers };
export default theme;
