import React, { useEffect, useState } from "react";
import classnames from "classnames";
import { t, Trans } from "@lingui/macro";

import { Controller, useForm } from "react-hook-form";
import { Button, Intent, Callout } from "@blueprintjs/core";
import { Dialog, Classes } from "@blueprintjs/core";
import { FormGroup, InputGroup } from "@blueprintjs/core";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";

import { showToast, getSuccessToast, getErrorToast } from "toaster";
import { useInputParams } from "hooks/use-input-params";
import { PROJECT_PROCESSING } from "constants/routes";
import { useGoTo } from "hooks/use-go-to";
import { useTheme } from "hooks/use-theme";
import { setTestId } from "test-utils/set-testid";
import { useDisclosure } from "hooks/use-disclosure";
import { gql, useApolloClient } from "@apollo/client";
import { CheckboxGroup } from "../checkbox-group";
import UploadGeotiff from "components/upload-geotiff";

import { createAoisFromGeometry } from "components/draw-rectangle-button";
import { DateInput2 } from "@blueprintjs/datetime2";
import { DATE_FNS_FORMATS } from "shared/date/date";
import { postRasters } from "shared/api/client/rasters";
import { MONTHS, WEEKDAYS } from "shared/date/daterange-shortcuts";

export const CREATE_PROCESSING = gql`
  mutation createProcessing($data: CreateProcessingInput!) {
    createProcessing(data: $data) {
      id
      name
    }
  }
`;

export const GET_DATA_PROVIDERS = gql`
  query getDataProviders {
    dataProviders {
      id
      displayName
      name
      urlTemplate
    }
  }
`;

const getDataProviders = (client) => async (variables) => {
  const result = await client.query({
    query: GET_DATA_PROVIDERS,
    fetchPolicy: "no-cache",
    variables: variables,
  });

  return result?.data?.dataProviders;
};

const createProcessing = (client) => async (variables) => {
  const result = await client.mutate({
    mutation: CREATE_PROCESSING,
    fetchPolicy: "no-cache",
    variables: { data: variables },
  });
  return result?.data?.createProcessing;
};

function CreateProcessingDialog({
  children,
  workflowName,
  workflowDesc,
  workflowDefId,
  defaultIsOpen = false,
  blocks,
}) {
  const [isGeoTiffDataProvider, setIsGeoTiffDataProvider] = useState(false);
  const [tiffAoi, setTiffAoi] = useState(null);
  const [rasterLoading, setRasterLoading] = useState(false);
  const fieldsConfig = {
    name: {
      maxLength: 40,
      required: true,
    },
    desc: {
      maxLength: 120,
    },
    dataProviderId: {
      required: true,
    },
    file: { required: isGeoTiffDataProvider },
    imageDate: { required: isGeoTiffDataProvider },
    imageId: { required: isGeoTiffDataProvider },
  };

  const queryClient = useQueryClient();

  const { projectId } = useParams();
  const goToProcessing = useGoTo(PROJECT_PROCESSING);

  const dialogStatus = useDisclosure(defaultIsOpen);

  const { themeClassName } = useTheme();
  const {
    register,
    control,
    handleSubmit,
    resetField,
    formState: { errors },
  } = useForm({ mode: "onChange" })

  const nameParams = useInputParams(errors.name, {
    required: t`This field is required`,
    maxLength: t`The name field may not be greater than ${fieldsConfig.name.maxLength} characters`,
  });

  const fileParams = useInputParams(errors.file, {
    required: t`This field is required`,
  });

  const descParams = useInputParams(errors.desc, {
    maxLength: t`The description field may not be greater than ${fieldsConfig.desc.maxLength} characters`,
  });

  const imageDateParams = useInputParams(errors.imageDate, {
    required: t`This field is required`,
  });

  const imageIdParams = useInputParams(errors.imageId, {
    required: t`This field is required`,
    // maxLength: t`The name field may not be greater than ${fieldsConfig.imageId.maxLength} characters`,
  });

  const dataProvidersQuery = useQuery({
    queryKey: ["data-providers"],
    queryFn: () => getDataProviders(client)(),
    select: (providers) => {
      return providers
        .filter((provider) => provider.urlTemplate !== null)
        .sort((a, b) =>
          (a.displayName || a.name).localeCompare(b.displayName || b.name),
        );
    },
    initialData: [],
  });

  const client = useApolloClient();

  const mutation = useMutation(createProcessing(client), {
    onSuccess: ({ id: processingId, name }) => {
      queryClient.invalidateQueries(["processings", projectId]);
      showToast(getSuccessToast(t`Processing ${name} successfully created`));
      if (tiffAoi) {
        aoiMutation.mutate({
          processingId: processingId,
          geometry: JSON.stringify(tiffAoi.geometry),
        });
        dialogStatus.onClose();
        return;
      }

      dialogStatus.onClose();
      goToProcessing({ projectId, processingId });
    },
    onError: () => showToast(getErrorToast(t`Error creating processing`)),
  });

  const aoiMutation = useMutation(createAoisFromGeometry(client), {
    refetchQueries: ["getAois"],
    onSuccess: (_, { processingId }) => {
      goToProcessing({ projectId, processingId });
    },
    onError: (e) => {
      console.error(e);
      showToast(getErrorToast(t`Error creating AOI`));
    },
  });

  useEffect(() => {
    if (dialogStatus.isOpen) {
      if (dataProvidersQuery.data?.[0]?.name === "GTIFF") {
        setIsGeoTiffDataProvider(true);
      }
      else {
        setIsGeoTiffDataProvider(false);
      }
    }
  }, [dialogStatus.isOpen, dataProvidersQuery.data]);

  const onSubmit = async (formData) => {
    const {
      name,
      desc,
      processingBlocks,
      dataProviderId,
      file,
      imageDate,
      imageId,
    } = formData;

    const mutatedBlocks = blocks.map((item) => ({
      name: item.name,
      enabled: processingBlocks?.some((block) => item.name === block),
    }));

    const payload = {
      projectId,
      workflowDefId,
      name,
      description: desc,
      blocks: mutatedBlocks,
    };
    if (dataProviderId) Object.assign(payload, { dataProviderId });

    if (file) {
      const formdata = new FormData();
      formdata.append("file", file[0]);
      if (tiffAoi) {
        try {
          setRasterLoading(true);
          const data = await postRasters(formdata);
          Object.assign(payload, { url: data.url, sourceType: "local" });
          const metaData = {
            image_id: imageId,
            image_date: imageDate,
          };
          Object.assign(payload, { meta: JSON.stringify(metaData) });
          setRasterLoading(false);
        } catch (error) {
          setRasterLoading(false);
          showToast(
            getErrorToast(
              t`File upload failed. Please try again later or contact us.`,
            ),
          );
          console.error(error);
          return;
        }
      }
    }
    mutation.mutate(payload);
  };

  const registerField = (elName, options) => {
    const { onChange, onBlur, name, ref } = register(elName, options);

    return { onChange, onBlur, name, inputRef: ref };
  };

  const handleSelectChange = (e) => {
    const currentDataProvider = dataProvidersQuery.data.find(
      (item) => item.id === e.target.value,
    );
    if (currentDataProvider.name === "GTIFF") {
      setIsGeoTiffDataProvider(true);
    } else if (currentDataProvider.name !== "GTIFF" && isGeoTiffDataProvider) {
      setIsGeoTiffDataProvider(false);
    }
  };

  return (
    <>
      <Dialog
        className={classnames(themeClassName, "create-processing-dialog")}
        title={<Trans>Create processing</Trans>}
        isOpen={dialogStatus.isOpen}
        onClose={dialogStatus.onClose}
      >
        <form onSubmit={handleSubmit(onSubmit)}>
          <div className={Classes.DIALOG_BODY}>
            <Callout className="workflow-info" title={workflowName}>
              {workflowDesc}
            </Callout>
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
                intent={nameParams.intent}
                disabled={mutation.isLoading}
                {...registerField("name", fieldsConfig.name)}
              />
            </FormGroup>
            <FormGroup
              label={<Trans>Description</Trans>}
              labelFor="desc"
              labelInfo={<Trans>(optional)</Trans>}
              helperText={<Trans id={descParams.helper} />}
              intent={descParams.intent}
            >
              <InputGroup
                large
                id="desc"
                type="text"
                intent={descParams.intent}
                disabled={mutation.isLoading}
                {...registerField("desc", fieldsConfig.desc)}
              />
            </FormGroup>
            <FormGroup
              label={<Trans>Data Provider</Trans>}
              labelFor="workflowId"
              helperText={<Trans id={descParams.helper} />}
              intent={descParams.intent}
            >
              <div className="bp4-html-select" style={{ width: "100%" }}>
                <select
                  id="workflowId"
                  disabled={dataProvidersQuery.isLoading}
                  {...register("dataProviderId", fieldsConfig.dataProviderId)}
                  onChange={handleSelectChange}
                >
                  {dataProvidersQuery.data &&
                    dataProvidersQuery.data.map(({ displayName, name, id }) => {
                      return (
                        <option key={id} value={id}>
                          {displayName || name}
                        </option>
                      );
                    })}
                </select>
                <span className="bp4-icon bp4-icon-double-caret-vertical"></span>
              </div>
            </FormGroup>

            {isGeoTiffDataProvider && (
              <UploadGeotiff
                register={register}
                fieldsConfig={fieldsConfig}
                fileParams={fileParams}
                setTiffAoi={setTiffAoi}
                resetField={resetField}
              />
            )}

            {isGeoTiffDataProvider && (
              <FormGroup
                label={<Trans>Дата съемки</Trans>}
                labelFor="imageDate"
                helperText={<Trans id={imageDateParams.helper} />}
                intent={imageDateParams.intent}
              >
                <Controller
                  name="imageDate"
                  control={control}
                  render={({ field }) => (
                    <DateInput2
                      id="imageDate"
                      {...registerField("imageDate", fieldsConfig.imageDate)}
                      {...DATE_FNS_FORMATS.yyyyMMdd}
                      placeholder="--/--/----"
                      value={field.value}
                      onChange={(date) => field.onChange(date)}
                      maxDate={new Date()}
                      dayPickerProps={{
                        months: MONTHS,
                        weekdaysShort: WEEKDAYS,
                      }}
                      popoverProps={{
                        usePortal: false,
                        position: "top",
                      }}
                    />
                  )}
                />
              </FormGroup>
            )}

            {isGeoTiffDataProvider && (
              <FormGroup
                label={<Trans>Идентификатор изображения</Trans>}
                labelFor="imageId"
                helperText={<Trans id={imageIdParams.helper} />}
                intent={imageIdParams.intent}
              >
                <InputGroup
                  large
                  id="imageId"
                  type="text"
                  intent={imageIdParams.intent}
                  disabled={mutation.isLoading}
                  {...registerField("imageId", fieldsConfig.imageId)}
                />
              </FormGroup>
            )}

            {blocks.length > 0 && (
              <CheckboxGroup
                name="processingBlocks"
                options={blocks}
                control={control}
                label={<Trans>Options</Trans>}
              />
            )}
          </div>
          <div className={Classes.DIALOG_FOOTER}>
            <div className={Classes.DIALOG_FOOTER_ACTIONS}>
              <Button
                elementRef={setTestId`cancel-create-processing`}
                intent={Intent.NONE}
                text={<Trans>Cancel</Trans>}
                onClick={dialogStatus.onClose}
                disabled={mutation.isLoading}
              />
              <Button
                elementRef={setTestId`submit-create-processing`}
                type="submit"
                intent={Intent.PRIMARY}
                text={<Trans id="Save" />}
                loading={
                  mutation.isLoading || rasterLoading || aoiMutation.isLoading
                }
              />
            </div>
          </div>
        </form>
      </Dialog>
      {typeof children === "function"
        ? children({
            showDialog: dialogStatus.onOpen,
            hideDialog: dialogStatus.onClose,
          })
        : children}
    </>
  );
}

export default React.memo(CreateProcessingDialog);
