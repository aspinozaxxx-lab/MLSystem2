import React from "react";
import classnames from "classnames";
import { t, Trans } from "@lingui/macro";
import { useForm } from "react-hook-form";
import { Dialog, Classes, Alignment, TextArea } from "@blueprintjs/core";
import { Button, Intent, Switch } from "@blueprintjs/core";
import { FormGroup, InputGroup } from "@blueprintjs/core";

import { useInputParams } from "hooks/use-input-params";
import { setTestId } from "test-utils/set-testid";
import { useTheme } from "hooks/use-theme";

export const fieldsConfig = {
  name: {
    maxLength: 40,
    required: true,
  },
  description: {
    maxLength: 120,
  },
};

function CreateProjectDialog({
  isOpen,
  handleClose,
  createProject,
  isLoading,
}) {
  const { themeClassName } = useTheme();
  const { register, handleSubmit, formState: { errors }} = useForm({ mode: "onChange" });

  const nameParams = useInputParams(errors.name, {
    required: t`This field is required`,
    maxLength: t`The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  });
  const descriptionParams = useInputParams(errors.description, {
    maxLength: t`The description field may not be greater than ${fieldsConfig.description.maxLength} characters`,
  });

  const registerField = (elName, options) => {
    const {onChange, onBlur, name, ref } = register(elName, options)

    return {onChange: onChange, onBlur: onBlur, name: name, inputRef: ref};
  }

  return (
    <Dialog
      className={classnames(themeClassName, "create-project-dialog")}
      title={<Trans id="Project creation" />}
      isOpen={isOpen}
      onClose={handleClose}
    >
      <form onSubmit={handleSubmit(createProject)}>
        <div className={Classes.DIALOG_BODY}>
          <FormGroup
            label={<Trans id="Name" />}
            labelFor="name"
            helperText={<Trans id={nameParams.helper} />}
            intent={nameParams.intent}
          >
            <InputGroup
              large
              autoFocus
              id="name"
              type="text"
              intent={nameParams.intent}
              disabled={isLoading}
              {...registerField("name", fieldsConfig.name)}
            />
          </FormGroup>
          <FormGroup
            label={<Trans id="Description" />}
            labelFor="description"
            labelInfo={<Trans id="(optional)" />}
            helperText={<Trans id={descriptionParams.helper} />}
            intent={descriptionParams.intent}
          >
            <TextArea
              fill
              large
              id="description"
              growVertically={true}
              {...registerField("description", fieldsConfig.description)}
              intent={descriptionParams.intent}
              disabled={isLoading}
            />
          </FormGroup>
          <FormGroup>
            <Switch
              large
              {...registerField("addDefaultWds", {})}
              label={<Trans id="Include default workflow definitions" />}
              disabled={isLoading}
              alignIndicator={Alignment.LEFT}
            />
          </FormGroup>
        </div>
        <div className={Classes.DIALOG_FOOTER}>
          <div className={Classes.DIALOG_FOOTER_ACTIONS}>
            <Button
              large
              elementRef={setTestId`cancel-create-project`}
              intent={Intent.NONE}
              text={<Trans id="Cancel" />}
              onClick={handleClose}
              disabled={isLoading}
            />
            <Button
              large
              elementRef={setTestId`submit-create-project`}
              type="submit"
              intent={Intent.PRIMARY}
              text={<Trans id="Save" />}
              loading={isLoading}
            />
          </div>
        </div>
      </form>
    </Dialog>
  );
}

export { CreateProjectDialog };
