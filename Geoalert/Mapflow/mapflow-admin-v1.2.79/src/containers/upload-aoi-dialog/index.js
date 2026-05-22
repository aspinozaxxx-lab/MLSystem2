import React, { useCallback, useMemo, useState } from "react";
import classnames from "classnames";
import { Trans, t } from "@lingui/macro";
import { useDropzone } from "react-dropzone";
import { gql, useMutation } from "@apollo/client";
import {
  Classes,
  Dialog,
  Text,
  Button,
  Intent,
  Callout,
} from "@blueprintjs/core";

import { useParams } from "react-router-dom";

import { useTheme } from "hooks/use-theme";
import { formatBytes } from "./format-bytes";
import { useAppToaster } from "hooks/use-app-toaster";
import { ToastTemplates } from "constants/toasts";
import { getBounds } from "utils/bounds-from-geometry";

export const CREATE_AOIS_FROM_FILE = gql`
  mutation createAoisFromFile(
    $processingId: ID!
    $file: Upload!
  ) {
    createAoisFromFile(
      data: {
        processingId: $processingId
        file: $file
      }
    ) {
      count
      bbox
    }
  }
`;


function UploadAoiDialog({ isOpen, openDialog, handleClose, mapAPI }) {
  const { processingId } = useParams();

  const { themeClassName } = useTheme();
  const [files, setFiles] = useState([]);

  const showToast = useAppToaster();

  const cleanUp = useCallback(() => {
    setFiles([]);
  }, []);

  const fitAoiBounds = useCallback(
    (geometry) => {
      try {
        if (mapAPI) mapAPI.fitBounds(getBounds(geometry), { padding: 50 });
      } catch (error) {
        console.error(error);
      }
    },
    [mapAPI],
  );

  const onCompleted = useCallback(
    ({ createAoisFromFile }) => {
      cleanUp();
      handleClose();
      showToast(getSuccessToast(createAoisFromFile.count));
      fitAoiBounds(createAoisFromFile.bbox);
    },
    [handleClose, showToast, cleanUp, fitAoiBounds],
  );
  const onError = useCallback(() => {
    const action = !isOpen ? { onClick: openDialog, text: "WTF?!" } : {};
    showToast(getErrorToast(action));
  }, [isOpen, openDialog, showToast]);

  const [handleUpload, { loading, error }] = useMutation(
    CREATE_AOIS_FROM_FILE,
    { refetchQueries: ["getAois"], onCompleted, onError },
  );

  const onDrop = useCallback(
    (acceptedFiles) => {
      const isBadExtention = acceptedFiles.some(
        ({ name }) => !(name.endsWith(".json") || name.endsWith(".geojson")),
      );
      if (isBadExtention) showToast(getBadExtentionToast());
      else setFiles(acceptedFiles);
    },
    [showToast],
  );

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject,
  } = useDropzone({
    onDrop,
    disabled: loading,
    multiple: false,
  });

  const onSubmit = useCallback(() => {
    const [file] = files;
    const variables = {
      processingId,
      file,
    };
    handleUpload({ variables });
  }, [processingId, files, handleUpload]);

  const isNotFiles = useMemo(() => files.length === 0, [files]);
  const dropzoneClassName = useMemo(
    () =>
      classnames(
        "dropzone",
        { "dropzone--active": isDragActive },
        { "dropzone--accept": isDragAccept || !isNotFiles },
        { "dropzone--reject": isDragReject || error },
        { "dropzone--disabled": loading },
      ),
    [isDragActive, isDragAccept, isNotFiles, isDragReject, error, loading],
  );

  const fileList = useMemo(
    () =>
      files.map(({ name, size }, i) => {
        return (
          <li key={i}>
            <div>{`Name: ${name}`}</div>
            <div>{`Size: ${formatBytes(size)}`}</div>
          </li>
        );
      }),
    [files],
  );

  const message = !isNotFiles ? (
    <ul>{fileList}</ul>
  ) : (
    <Trans id="dnd-file-message">
      Drag 'n' drop files here, or click to select files
    </Trans>
  );
  
  return (
    <Dialog
      className={classnames(themeClassName, "upload-aoi-dialog")}
      icon="info-sign"
      isOpen={isOpen}
      onClose={handleClose}
      title={<Trans>Upload files</Trans>}
      autoFocus={true}
      canEscapeKeyClose={true}
      canOutsideClickClose={true}
      enforceFocus={true}
      usePortal={true}
    >
      <div className={Classes.DIALOG_BODY}>
        <div className="upload-aoi-dialog__body">
          <div {...getRootProps({ className: dropzoneClassName })}>
            <input {...getInputProps()} />
            <Text>{message}</Text>
          </div>
          {error && (
            <Callout
              intent={Intent.DANGER}
              className="upload-aoi-dialog-error-message"
            >
              {JSON.stringify(error).substr(0, 300)}
            </Callout>
          )}
        </div>
      </div>
      <div className={classnames(Classes.DIALOG_FOOTER, "upload-controls")}>
        <Button
          large
          intent={Intent.SUCCESS}
          text={"Upload"}
          loading={loading}
          disabled={isNotFiles}
          onClick={onSubmit}
        />
      </div>
    </Dialog>
  );
}

export default React.memo(UploadAoiDialog);

const getBadExtentionToast = () => ({
  message: t`Bad extention, accepts only *.geojson/*.json files`,
  template: ToastTemplates.ERROR,
});

const getSuccessToast = (count) => ({
  message: t`Successfully loaded ${count} features`,
  template: ToastTemplates.SUCCESS,
});

const getErrorToast = (action) => ({
  message: t`Error while uploading aoi's`,
  template: ToastTemplates.ERROR,
  action,
});
