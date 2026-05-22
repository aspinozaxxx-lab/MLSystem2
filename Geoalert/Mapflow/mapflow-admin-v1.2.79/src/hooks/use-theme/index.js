import { Classes } from "@blueprintjs/core";
import { gql, useQuery, useMutation } from "@apollo/client";

import { ThemeNames } from "constants/common";

const DARK_THEME = Classes.DARK;
const LIGHT_THEME = "";

export const Themes = {
  light: LIGHT_THEME,
  dark: DARK_THEME,
};

export const GET_THEME = gql`
  query getTheme {
    theme @client
  }
`;
export const TOGGLE_THEME = gql`
  mutation toggleTheme {
    toggleTheme @client
  }
`;

export function useTheme(initialTheme = ThemeNames.LIGHT) {
  const { data = {} } = useQuery(GET_THEME);
  const theme = data.theme || initialTheme;
  const [toggleTheme] = useMutation(TOGGLE_THEME);
  return { theme, toggleTheme, themeClassName: Themes[theme] };
}
