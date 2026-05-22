import React from "react";

export let _handleClose,
  _createProject,
  _isLoading = jest.fn(),
  _isOpen = jest.fn();

function CreateProjectDialog({
  isOpen,
  isLoading,
  handleClose,
  createProject,
}) {
  _handleClose = jest.fn(handleClose);
  _createProject = jest.fn(createProject);

  React.useEffect(() => {
    _isLoading(isLoading);
  }, [isLoading]);
  React.useEffect(() => {
    _isOpen(isOpen);
  }, [isOpen]);
  return null;
}

export { CreateProjectDialog };
