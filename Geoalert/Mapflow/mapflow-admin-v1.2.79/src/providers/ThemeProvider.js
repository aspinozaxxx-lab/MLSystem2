import React from "react";
import { useTheme } from "hooks/use-theme";

export const ThemeProvider = ({ children, initialTheme }) => {
  const { themeClassName } = useTheme(initialTheme);
  return (
    <div style={{ height: "100%", width: "100%" }} className={themeClassName}>
      {children}
    </div>
  );
};

export default React.memo(ThemeProvider);
