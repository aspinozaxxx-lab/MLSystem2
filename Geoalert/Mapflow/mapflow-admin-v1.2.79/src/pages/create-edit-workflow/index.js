import React, { useState, useCallback, useRef } from "react";
import { gql, useApolloClient } from "@apollo/client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Controller, useForm } from "react-hook-form";
import { useParams } from "react-router-dom";
import { t, Trans } from "@lingui/macro";
import {
  H2,
  Button,
  Intent,
  FormGroup,
  InputGroup,
  FileInput,
  TextArea,
  Switch,
} from "@blueprintjs/core";
import pathOr from "ramda/src/pathOr";

import { Breadcrumbs } from "containers";

import { useInputParams } from "hooks/use-input-params";
import { useGoTo } from "hooks/use-go-to";

import { WORKFLOWS } from "constants/routes";
import { GET_PROJECT_WORKFLOWS } from "components/project-workflow-list";
import { setTestId } from "test-utils/set-testid";
import { showToast, getSuccessToast, getErrorToast } from "toaster";

import { IconNames } from "@blueprintjs/icons";
import { WorkflowEditor } from "components/workflow-editor";

export const GET_WORKFLOW = gql`
  query getWorkflowDef($id: ID!) {
    workflowDef(id: $id) {
      id
      name
      description
      yml
      created
      updated
      archived
      isDefault
    }
  }
`;

export const UPDATE_WORKFLOW = gql`
  mutation updateWorkflowDef($data: UpdateWorkflowDefInput!) {
    updateWorkflowDef(data: $data) {
      id
      name
      description
    }
  }
`;

export const CREATE_WORKFLOW = gql`
  mutation createWorkflowDef($data: CreateWorkflowDefInput!) {
    createWorkflowDef(data: $data) {
      id
      name
      description
    }
  }
`;

const fieldsConfig = {
  name: { maxLength: 40, required: true },
  description: { maxLength: 120 },
  file: { required: false },
  yml: { required: true, minLength: 5 },
  isDefault: { required: false },
};

function CreateEditWorkflow() {
  const { workflowDefId } = useParams();

  const editorRef = useRef({});

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    setValue,
    control,
  } = useForm({
    mode: "onChange",
  });

  const { data: workflow, isLoading: workflowLoading } = useQuery({
    queryKey: ["getWorkflowDef", workflowDefId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_WORKFLOW,
        fetchPolicy: "no-cache",
        variables: { id: workflowDefId },
      });
      return result?.data?.workflowDef;
    },
    onSuccess: (data) => {
      setValue("name", data?.name);
      setValue("description", data?.description);
      setValue("isDefault", data?.isDefault);
      setValue("yml", data?.yml);
    },
    enabled: !!workflowDefId && !isDirty,
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

  const goToWorkflowsPage = useGoTo(WORKFLOWS);

  const client = useApolloClient();
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationKey: ["updateWorkflowDef", workflowDefId],
    mutationFn: async (data) => {
      const result = await client.mutate({
        mutation: UPDATE_WORKFLOW,
        variables: { data },
        context: { hasUpload: true },
      });
      return result?.data?.updateWorkflowDef;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["workflowsDefs"],
        force: true,
      });
      queryClient.invalidateQueries({
        queryKey: ["workflowDef", "name"],
        force: true,
      });
      showToast(
        getSuccessToast(t`Workflow "${workflow.name}" updated`, {
          icon: IconNames.EDIT,
        }),
      );
      goToWorkflowsPage();
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error updating workflow`));
    },
    enabled: !!workflowDefId && !!workflow,
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
      goToWorkflowsPage();
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error creating workflow`));
    },
    refetchQueries: [{ query: GET_PROJECT_WORKFLOWS }],
  });

  const onSubmit = (d) => {
    const { name, description, isDefault } = d;

    editorRef.current.save({
      onError: (message) => showToast(getErrorToast(message)),
      onSuccess: (yml) => {
        if (workflowDefId) {
          updateMutation.mutate({
            id: workflowDefId,
            description,
            name,
            ymlString: yml,
            pricePerSqKm: 0,
            isDefault,
          });
        } else {
          createMutation.mutate({
            description,
            name,
            ymlString: yml,
            pricePerSqKm: 0,
            isDefault,
          });
        }
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
        const r = reader.result ?? "";
        setValue("yml", r);
        editorRef.current.updateWithExtractWd(r);
      };

      reader.readAsText(file);
    },
    [setValue],
  );

  const registerField = (elName, options) => {
    const { onChange, onBlur, name, ref } = register(elName, options);

    return { onChange: onChange, onBlur: onBlur, name: name, inputRef: ref };
  };

  const isLoading =
    updateMutation.isLoading ||
    createMutation.isLoading ||
    (workflowDefId && workflowLoading);

  return (
    <div className="create-edit-workflow">
      <Breadcrumbs />

      <div className="create-edit-workflow__container">
        <H2>
          {workflowDefId ? (
            <Trans>Edit Workflow</Trans>
          ) : (
            <Trans>Create a Workflow</Trans>
          )}
        </H2>
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
                buttonText={t`Browse`}
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
              elementRef={setTestId`cancel-create-workflow`}
              text={<Trans>Cancel</Trans>}
              disabled={isLoading}
              onClick={goToWorkflowsPage}
            />
            <Button
              large
              type="submit"
              intent={Intent.PRIMARY}
              elementRef={setTestId`submit-create-workflow`}
              text={<Trans>Save</Trans>}
              disabled={isLoading}
              loading={isLoading}
            />
          </div>
        </form>
      </div>
    </div>
  );
}

export default React.memo(CreateEditWorkflow);
