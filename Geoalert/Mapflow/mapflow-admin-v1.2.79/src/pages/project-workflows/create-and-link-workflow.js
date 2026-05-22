import { useApolloClient } from "@apollo/client";
import {
  Button,
  Callout,
  Dialog,
  DialogBody,
  FileInput,
  FormGroup,
  H2,
  InputGroup,
  Intent,
  Switch,
  Text,
  TextArea,
} from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { t, Trans } from "@lingui/macro";
import { useMutation } from "@tanstack/react-query";
import classNames from "classnames";
import { GET_PROJECT_WORKFLOWS } from "components/project-workflow-list";
import useProjectQuery from "components/project-workflow-list/queries";
import { WorkflowEditor } from "components/workflow-editor";
import { useInputParams } from "hooks/use-input-params";
import { useTheme } from "hooks/use-theme";
import { CREATE_WORKFLOW } from "pages/create-edit-workflow";
import { LINK_WORKFLOW } from "pages/link-workflow";
import { LINK_WORKFLOW_TO_USER } from "pages/manage-workflow-users";
import { pathOr } from "ramda";
import React, { useCallback, useRef, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { useParams } from "react-router-dom";
import { getErrorToast, getSuccessToast, showToast } from "toaster";

const fieldsConfig = {
  name: { maxLength: 40, required: true },
  description: { maxLength: 120 },
  file: { required: false },
  yml: { required: true, minLength: 5 },
  isDefault: { required: false },
  pricePerSqKm: {
    min: 0,
    required: false,
    onChange: (e) => {
      const value = e.target.value;
      if (value === "") {
        e.target.value = 0;
        return;
      }
      e.target.value = value.replace(/[^0-9.]/g, "").replace(/(\..*)\./g, "$1");
    },
  },
};

function CreateAndLinkWorkflow() {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const { projectId } = useParams();

  const toggleDialog = () => setIsDialogOpen((p) => !p);
  const closeDialog = () => setIsDialogOpen(false);

  const editorRef = useRef({});

  const {
    control,
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    reset,
  } = useForm({
    mode: "onChange",
  });

  const nameParams = useInputParams(errors.name, {
    required: t`This field is required`,
    maxLength: t`The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  });
  const descriptionParams = useInputParams(errors.description, {
    maxLength: t`The description field may not be greater than ${fieldsConfig.description.maxLength} characters`,
  });
  const fileParams = useInputParams(errors.file, {
    required: t`This field is required`,
  });
  const ymlParams = useInputParams(errors.yml, {
    required: t`This field is required`,
  });

  const client = useApolloClient();

  const { data, isLoading: userLoading } = useProjectQuery(projectId);
  const userId = data?.userId;

  const linkToProjectMutation = useMutation({
    mutationKey: ["linkWorkflowDefToProject", projectId],
    mutationFn: async ({ workflowDefId, projectId }) => {
      const result = await client.mutate({
        mutation: LINK_WORKFLOW,
        variables: { workflowDefId, projectId },
      });
      return result?.data?.linkWorkflowDefToProject;
    },
    onSuccess: (data) => {
      showToast(
        getSuccessToast(
          t`Workflow linked to this project and Default project.`,
          {
            timeout: 10000,
            isCloseButtonShown: true,
          },
        ),
      );
    },
    onError: (e) => {
      showToast(getErrorToast(t`Error linking workflow to project`));
    },
  });

  const linkToUserMutation = useMutation({
    mutationKey: ["linkWorkflowDefToUser", projectId],
    mutationFn: async ({ workflowDefId, userId }) => {
      const result = await client.mutate({
        mutation: LINK_WORKFLOW_TO_USER,
        variables: { workflowDefId, userId },
      });

      return result?.data?.linkWorkflowDefToUser;
    },
    onSuccess: (data, payload) => {
      showToast(getSuccessToast(t`Workflow linked to user`));
      linkToProjectMutation.mutate({
        workflowDefId: payload.workflowDefId,
        projectId,
      });
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error linking Workflow to user`));
    },
  });

  const createMutation = useMutation({
    mutationKey: ["createWorkflowDef"],
    mutationFn: async (data) => {
      const result = await client.mutate({
        mutation: CREATE_WORKFLOW,
        fetchPolicy: "no-cache",
        variables: { data },
        context: { hasUpload: true },
      });

      return result?.data?.createWorkflowDef;
    },
    onSuccess: (data) => {
      showToast(
        getSuccessToast(t`Workflow "${data.name}" successfully created`),
      );
      linkToUserMutation.mutate({
        workflowDefId: data.id,
        userId,
      });

      reset();
      closeDialog();
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error creating workflow`));
    },
    refetchQueries: [{ query: GET_PROJECT_WORKFLOWS }],
  });

  const onSubmit = (d) => {
    const { name, description, isDefault, pricePerSqKm } = d;

    editorRef.current.save({
      onError: (message) => showToast(getErrorToast(message)),
      onSuccess: (yml) => {
        createMutation.mutate({
          description,
          name,
          ymlString: yml,
          pricePerSqKm: parseFloat(pricePerSqKm),
          isDefault,
        });
      },
    });
  };

  const [filename, setFilename] = useState(null);
  const onFileInputChange = useCallback(
    (d) => {
      const filename = pathOr(null, ["target", "files", "0", "name"], d);
      const file = pathOr(null, ["target", "files", "0"], d);

      setFilename(filename);

      const reader = new FileReader();
      reader.onload = (e) => {
        setValue("yml", reader.result || null);
      };

      reader.readAsText(file);
    },
    [setValue],
  );

  const registerField = (elName, options) => {
    const { onChange, onBlur, name, ref } = register(elName, options);

    return { onChange: onChange, onBlur: onBlur, name: name, inputRef: ref };
  };

  const isLoading = createMutation.isLoading || userLoading;

  const { themeClassName } = useTheme();

  return (
    <div className="create-and-link">
      <Button
        outlined
        minimal
        large
        icon={IconNames.APPLICATION}
        intent={Intent.PRIMARY}
        text={<Trans id="Create Workflow" />}
        disabled={isLoading}
        onClick={toggleDialog}
      />

      <Dialog
        className={classNames(themeClassName, "create-and-link__dialog")}
        isOpen={isDialogOpen}
        onClose={closeDialog}
      >
        <DialogBody style={{ oveflow: "hidden" }}>
          <div className="create-and-link__container">
            <H2>
              <Trans>Create Workflow</Trans>
            </H2>

            <Callout intent={Intent.PRIMARY}>
              <Text>
                {t`THIS ACTION WILL LINK THE NEW WORKFLOW TO THE OWNER OF THIS PROJECT`}
              </Text>
            </Callout>

            <form
              className="create-workflow-form"
              autoComplete="off"
              onSubmit={handleSubmit(onSubmit)}
            >
              <div className="create-workflow-form__body">
                <FormGroup
                  label={<Trans>Name</Trans>}
                  labelFor="name"
                  helperText={<Trans id={nameParams.helper} />}
                  intent={nameParams.intent}
                >
                  <InputGroup
                    large
                    autoFocus
                    id="name"
                    type="text"
                    name="name"
                    intent={nameParams.intent}
                    disabled={isLoading}
                    {...registerField("name", fieldsConfig.name)}
                  />
                </FormGroup>

                <FormGroup
                  label={<Trans>Description</Trans>}
                  labelFor="description"
                  helperText={<Trans id={descriptionParams.helper} />}
                  intent={descriptionParams.intent}
                >
                  <TextArea
                    fill
                    large
                    id="description"
                    name="description"
                    growVertically={true}
                    {...registerField("description", fieldsConfig.description)}
                    intent={descriptionParams.intent}
                    disabled={isLoading}
                  />
                </FormGroup>

                <FormGroup
                  label={<Trans>Default</Trans>}
                  labelFor="isDefault"
                  className="create-workflow-form__default"
                >
                  <Switch
                    id="isDefault"
                    name="isDefault"
                    {...registerField("isDefault", fieldsConfig.isDefault)}
                  />
                </FormGroup>

                <FormGroup
                  label={<Trans>Workflow definition</Trans>}
                  labelFor="file"
                  helperText={<Trans id={fileParams.helper} />}
                  intent={fileParams.intent}
                >
                  <FileInput
                    fill
                    large
                    hasSelection={filename}
                    onInputChange={onFileInputChange}
                    text={filename ? filename : <Trans>Choose file...</Trans>}
                    inputProps={{
                      id: "file",
                      name: "file",
                      intent: fileParams.intent,
                      ...register("file", fieldsConfig.file),
                    }}
                    disabled={isLoading}
                  />
                </FormGroup>

                <FormGroup
                  labelFor="ymlEditor"
                  helperText={<Trans id={ymlParams.helper} />}
                  intent={ymlParams.intent}
                >
                  <Controller
                    name="yml"
                    control={control}
                    render={({ field: { value, onChange } }) => {
                      return (
                        <WorkflowEditor
                          editorRef={editorRef}
                          id="workflow-definition"
                          placeholder={t`Please enter workflow definition in YAML`}
                          disabled={isLoading}
                          value={value}
                          onChange={onChange}
                        />
                      );
                    }}
                  />
                </FormGroup>
              </div>

              <div className="create-workflow-form__actions">
                <Button
                  large
                  intent={Intent.NONE}
                  text={<Trans>Cancel</Trans>}
                  disabled={isLoading}
                  onClick={closeDialog}
                />
                <Button
                  large
                  type="submit"
                  intent={Intent.PRIMARY}
                  text={<Trans>Save</Trans>}
                  disabled={isLoading}
                  loading={isLoading}
                />
              </div>
            </form>
          </div>
        </DialogBody>
      </Dialog>
    </div>
  );
}

export default CreateAndLinkWorkflow;
