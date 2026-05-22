import React, { useState } from "react";
import { Alert } from "@blueprintjs/core";
import { useTheme } from "hooks/use-theme";

function ConfirmDialog({
  children,
  className,
  icon,
  intent,
  confirmButtonText,
  cancelButtonText,
  onConfirm,
  text,
  ...alertProps
}) {
  const [isOpen, setIsOpen] = useState(false);
  const handleOpen = (e) => {
    e.stopPropagation();
    setIsOpen(true);
  };
  const handleClose = () => setIsOpen(false);

  const { themeClassName } = useTheme();

  return (
    <div className={className} onClick={(e) => e.stopPropagation()}>
      <Alert
        {...alertProps}
        className={themeClassName}
        isOpen={isOpen}
        icon={icon}
        intent={intent}
        confirmButtonText={confirmButtonText}
        cancelButtonText={cancelButtonText}
        onCancel={handleClose}
        onConfirm={() => onConfirm(handleClose)}
      >
        {text}
      </Alert>
      {typeof children === "function"
        ? children({
            showDialog: handleOpen,
            hideDialog: handleClose,
          })
        : children}
    </div>
  );
}

export default React.memo(ConfirmDialog);
