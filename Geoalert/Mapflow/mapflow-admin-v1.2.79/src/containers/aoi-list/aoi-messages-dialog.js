import React, { memo } from "react";

import classnames from "classnames";
import { Trans } from "@lingui/macro";
import { Classes, Dialog } from "@blueprintjs/core";

import { useTheme } from "hooks/use-theme";
import { Icon } from "@blueprintjs/core";
import { uniq } from "ramda";

function AoiMessagesDialog({ messages, isOpen, handleClose }) {
  const { themeClassName } = useTheme();

  if (!messages || !messages.length) {
    return null;
  }

  return (
    <div
      className="aoi-messages-container"
      onClick={(e) => e.stopPropagation()}
    >
      <Dialog
        className={classnames(themeClassName, "upload-aoi-dialog")}
        icon="info-sign"
        isOpen={isOpen}
        onClose={(event) => {
          event.stopPropagation();
          handleClose(false);
        }}
        title={<Trans>Messages</Trans>}
        autoFocus={true}
        canEscapeKeyClose={true}
        canOutsideClickClose={true}
        enforceFocus={true}
        usePortal={true}
      >
        <div className={Classes.DIALOG_BODY}>
          <div className="upload-aoi-dialog__body aoi-messages">
            <code>
              {uniq(messages).map(({ message }) => (
                <div key={message}>
                  <span>
                    {
                      "Произошла ошибка обработки, обратитесь к администратору. Детали: ("
                    }
                  </span>
                  &gt;&gt;&gt; <span>{message}</span>
                  <span>{")"}</span>
                  <br />
                  <br />
                </div>
              ))}
            </code>
          </div>
        </div>
      </Dialog>
    </div>
  );
}
AoiMessagesDialog.displayName = "AoiMessagesDialog";

export default memo(AoiMessagesDialog);
