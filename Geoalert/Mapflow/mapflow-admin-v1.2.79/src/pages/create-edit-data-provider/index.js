import React from "react";
import { useApolloClient } from "@apollo/client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { useParams } from "react-router-dom";
import { t, Trans } from "@lingui/macro";
import {
  H2,
  Button,
  Intent,
  FormGroup,
  InputGroup,
  Switch,
} from "@blueprintjs/core";
import { Breadcrumbs } from "containers";

import { useInputParams } from "hooks/use-input-params";
import { useGoTo } from "hooks/use-go-to";

import { DATA_PROVIDERS } from "constants/routes";
import { setTestId } from "test-utils/set-testid";
import { showToast, getSuccessToast, getErrorToast } from "toaster";

import { IconNames } from "@blueprintjs/icons";
import {
  CREATE_DATA_PROVIDER,
  GET_DATA_PROVIDER,
  UPDATE_DATA_PROVIDER,
} from "components/data-provider/queries";
import StateLoading from "components/state-loading";
import ErrorMessage from "components/error-message";

const fieldsConfig = {
  name: { maxLength: 40, required: true },
  displayName: { maxLength: 120, required: false },
  credentialsUsername: { required: false },
  credentialsPassword:  required: false },
  credentialsToken:  required: false },
  urlTemplate: { required: false },
  previewUrl: { required: false },
  isDefault: { required: false },
  mapfileUri: { required: false },
};

function CreateEditWorkflow() {
  const { dataProviderId } = useParams();

  const {
    data: dataProvidersResult,
    isLoading: dataProvidersLoading,
    isError: dataProvidersError,
  } = useQuery({
    queryKey: ["getDataProvider", dataProviderId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_DATA_PROVIDER,
        fetchPolicy: "no-cache",
        variables: { id: dataProviderId },
      });
      return result?.data.dataProviders[0];
    },
    onSuccess: (data) => {
      const {
        name,
        displayName,
        urlTemplate,
        previewUrl,
        isDefault,
        credentialsUsername,
        credentialsPassword,
        credentialsToken,
        mapfileUri,
      } = data;
      setValue("name", name);
      setValue("displayName", displayName);
      setValue("isDefault", isDefault);
      setValue("previewUrl", previewUrl);
      setValue("urlTemplate", urlTemplate);
      setValue("credentialsUsername", credentialsUsername);
      setValue("credentialsToken", credentialsToken);
      setValue("credentialsPassword", credentialsPassword);
      setValue("mapfileUri", mapfileUri);
    },
    enabled: !!dataProviderId,
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
  } = useForm({
    mode: "onChange",
  });

  const nameParams = useInputParams(errors.name, {
    required: t`This field is required`,
    maxLength: t`The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  });

  const displayNameParams = useInputParams(errors.displayName, {
    maxLength: t`The display name field may not be greater than ${fieldsConfig.displayName.maxLength} characters`,
  });

  const goToDataProvidersPage = useGoTo(DATA_PROVIDERS);

  const client = useApolloClient();
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationKey: ["updateDataProvider", dataProviderId],
    mutationFn: async (data) => {
      const result = await client.mutate({
        mutation: UPDATE_DATA_PROVIDER,
        variables: { data },
        context: { hasUpload: true },
      });
      return result?.data.dataProviders;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["dataProviders"],
        force: true,
      });
      queryClient.invalidateQueries({
        queryKey: ["dataProviders", "name"],
        force: true,
      });
      showToast(
        getSuccessToast(
          t`Data provider "${dataProvidersResult?.name}" updated`,
          {
            icon: IconNames.EDIT,
          },
        ),
      );
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error updating data provider`));
    },
    enabled: !!dataProviderId && !!dataProvidersResult,
  });

  const createMutation = useMutation({
    mutationKey: ["createDataProvider"],
    mutationFn: async (data) => {
      const result = await client.mutate({
        mutation: CREATE_DATA_PROVIDER,
        fetchPolicy: "no-cache",
        variables: { data },
        context: { hasUpload: true },
      });
      return result?.data;
    },
    onSuccess: (data) => {
      showToast(
        getSuccessToast(
          t`Data provider "${data?.createDataProvider.name}" successfully created`,
        ),
      );
      goToDataProvidersPage();
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error creating data provider`));
    },
  });

  const onSubmit = (values) => {
    const {
      name,
      displayName,
      urlTemplate,
      previewUrl,
      isDefault,
      credentialsUsername,
      credentialsPassword,
      credentialsToken,
      mapfileUri,
    } = values;

    if (dataProviderId) {
      updateMutation.mutate({
        id: dataProviderId,
        name,
        displayName,
        urlTemplate,
        previewUrl,
        isDefault,
        pricePerMp: 0,
        credentialsUsername,
        credentialsPassword,
        credentialsToken,
        mapfileUri,
      });
    } else {
      createMutation.mutate({
        name,
        displayName,
        urlTemplate,
        previewUrl,
        isDefault,
        pricePerMp: 0,
        credentialsUsername,
        credentialsPassword,
        credentialsToken,
        mapfileUri,
      });
    }
  };

  const registerField = (elName, options) => {
    const { onChange, onBlur, name, ref } = register(elName, options);

    return { onChange: onChange, onBlur: onBlur, name: name, inputRef: ref };
  };

  const isLoading =
    updateMutation.isLoading ||
    createMutation.isLoading ||
    (dataProviderId && dataProvidersLoading);

  if (dataProvidersLoading && dataProviderId) {
    return (
      <StateLoading
        style={{ flex: 1 }}
        title={<Trans id="Fetching data provider" />}
      />
    );
  }

  if (dataProvidersError) {
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Could not fetch data provider" />}
      />
    );
  }

  return (
    <div className="create-edit-workflow">
      <Breadcrumbs />

      <div className="create-edit-workflow__container">
        <H2>
          {dataProviderId ? (
            <Trans>Edit Data Provider</Trans>
          ) : (
            <Trans>Create Data Provider</Trans>
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
              label={<Trans>Display Name</Trans>}
              labelFor="displayName"
              helperText={<Trans id={displayNameParams.helper} />}
              intent={displayNameParams.intent}
            >
              <InputGroup
                large
                id="displayName"
                type="text"
                name="displayName"
                intent={displayNameParams.intent}
                disabled={isLoading}
                {...registerField("displayName", fieldsConfig.displayName)}
              />
            </FormGroup>

            <FormGroup
              label={<Trans>URL Template</Trans>}
              labelFor="urlTemplate"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="urlTemplate"
                type="text"
                name="urlTemplate"
                disabled={isLoading}
                {...registerField("urlTemplate", fieldsConfig.urlTemplate)}
              />
            </FormGroup>

            <FormGroup
              label={<Trans>Preview URL</Trans>}
              labelFor="previewUrl"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="previewUrl"
                type="text"
                name="previewUrl"
                disabled={isLoading}
                {...registerField("previewUrl", fieldsConfig.previewUrl)}
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
              label={<Trans>Credentials Username</Trans>}
              labelFor="credentialsUsername"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="credentialsUsername"
                type="text"
                name="credentialsUsername"
                disabled={isLoading}
                {...registerField(
                  "credentialsUsername",
                  fieldsConfig.credentialsUsername,
                )}
              />
            </FormGroup>

            <FormGroup
              label={<Trans>Credentials Password</Trans>}
              labelFor="credentialsPassword"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="credentialsPassword"
                type="text"
                name="credentialsPassword"
                disabled={isLoading}
                {...registerField(
                  "credentialsPassword",
                  fieldsConfig.credentialsPassword,
                )}
              />
            </FormGroup>

            <FormGroup
              label={<Trans>Credentials Token</Trans>}
              labelFor="credentialsToken"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="credentialsToken"
                type="text"
                name="credentialsToken"
                disabled={isLoading}
                {...registerField(
                  "credentialsToken",
                  fieldsConfig.credentialsToken,
                )}
              />
            </FormGroup>

            <FormGroup
              label={<Trans>Mapfile Источника</Trans>}
              labelFor="mapfileUri"
              intent={nameParams.intent}
            >
              <InputGroup
                large
                id="mapfileUri"
                type="text"
                name="mapfileUri"
                disabled={isLoading}
                {...registerField("mapfileUri", fieldsConfig.mapfileUri)}
              />
            </FormGroup>
          </div>

          <div className="create-workflow-form__actions">
            <Button
              large
              intent={Intent.NONE}
              elementRef={setTestId`cancel-create-dataProvider`}
              text={<Trans>Cancel</Trans>}
              disabled={isLoading}
              onClick={goToDataProvidersPage}
            />
            <Button
              large
              type="submit"
              intent={Intent.PRIMARY}
              elementRef={setTestId`submit-create-dataProvider`}
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
