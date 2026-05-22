import React, { useMemo } from "react";
import { Button } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

import { setTestId } from "test-utils/set-testid";
import { useLanguage } from "hooks/use-language";
import { useWindowWidth } from "hooks/use-window-width";

function LanguageButton({ minimal }) {
  const { language, toggleLanguage } = useLanguage();

  const width = useWindowWidth();
  const isTextVisible = useMemo(() => width > 450, [width]);

  return (
    <div>
      <Button
        icon={IconNames.TRANSLATE}
        text={isTextVisible && language}
        minimal={minimal}
        elementRef={setTestId`language-button`}
        onClick={toggleLanguage}
      />
    </div>
  );
}

export default React.memo(LanguageButton);
