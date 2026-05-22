import React, { useCallback, useState } from "react";
import { Trans, t } from "@lingui/macro";

import { FormGroup, FileInput } from "@blueprintjs/core";

import { useTiffReader } from "hooks/use-tiff-reader";
import { pathOr } from "ramda";
import { validateGeometry } from "lib/geojson-validation/validate-geometry";
import { getErrorToast, showToast } from "toaster";
import { parseTiffError } from "lib/geojson-validation/tif-error";

function UploadGeoTiff({
  fieldsConfig,
  register,
  fileParams,
  setTiffAoi,
  resetField,
}) {
  const [filename, setFilename] = useState(null);

  const { read } = useTiffReader({
    maxSizeLimit: 30000,
    validate: validateGeometry,
    onError: (error) => {
      resetField("file");
      setTiffAoi((prev) => null);
      setFilename(null);

      const textError = parseTiffError(error);
      if (textError) {
        showToast(getErrorToast(textError));
        return;
      }
      showToast(
        getErrorToast(t`The file doesn't match the required input parameters`),
      );
      console.error(error);
    },
    onSuccess: (tiff) => {
      setTiffAoi(tiff);
    },
  });

  const onChangeInputFile = useCallback(
    (e) => {
      const file = pathOr(null, ["target", "files"], e);
      const filename = pathOr(null, ["target", "files", "0", "name"], e);
      setFilename(filename);
      read(file);
    },
    [read],
  );

  return (
    <FormGroup
      label={<Trans>Upload Geotiff</Trans>}
      labelFor="file"
      helperText={fileParams.helper}
      intent={fileParams.intent}
    >
      <FileInput
        fill
        large
        hasSelection={"heyy"}
        onInputChange={onChangeInputFile}
        buttonText={t`Browse`}
        text={filename ? filename : <Trans>Choose file...</Trans>}
        inputProps={{
          id: "file",
          name: "file",
          accept: ".tif, .tiff",
          ...register("file", fieldsConfig.file),
        }}
      />
    </FormGroup>
  );
}

export default React.memo(UploadGeoTiff);
