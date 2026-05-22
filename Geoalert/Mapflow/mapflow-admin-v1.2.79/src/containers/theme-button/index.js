import React, { useMemo } from "react";

import { t, Trans } from "@lingui/macro";
import { Button, Classes } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

import { ThemeNames } from "constants/common";
import { setTestId } from "test-utils/set-testid";
import { useTheme } from "hooks/use-theme";
import { AppToaster } from "toaster";
import { useWindowWidth } from "hooks/use-window-width";

const Translates = {
  dark: t`You activated a light theme`,
  light: t`You activated a dark theme`,
};

function ThemeButton(minimal) {
  const { theme, toggleTheme } = useTheme();
  const icon = theme === ThemeNames.DARK ? IconNames.MOON : IconNames.FLASH;
  const toastIcon =
    theme !== ThemeNames.DARK ? IconNames.MOON : IconNames.FLASH;
  const toastClassName = theme !== ThemeNames.DARK ? Classes.DARK : "";

  const width = useWindowWidth();
  const isTextVisible = useMemo(() => width > 450, [width]);

  return (
    <div>
      <Button
        icon={icon}
        text={isTextVisible && <Trans id={ThemeNames.T[theme]} />}
        minimal={minimal}
        elementRef={setTestId`theme-button`}
        onClick={() => {
          AppToaster.show({
            icon: toastIcon,
            className: toastClassName,
            message: Translates[theme],
            timeout: 1500,
          });
          toggleTheme();
        }}
      />
    </div>
  );
}

export default React.memo(ThemeButton);
