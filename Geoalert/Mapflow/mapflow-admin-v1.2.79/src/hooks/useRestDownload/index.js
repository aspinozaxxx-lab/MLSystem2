import { useState } from "react";
import { AppToaster } from "toaster";
import { IconNames } from "@blueprintjs/icons";
import {
  getErrorToast,
  getPendingToast,
  getSuccessToast,
} from "hooks/use-mutation-with-toasts";
import { t } from "@lingui/macro";
import ky from "ky";
import { avanpostCookies } from "providers/auth-avanpost-provider";
import { BACKEND_URL } from "constants/__mocks__/envs";

export const useDownload = (url) => {
  const [loading, setLoading] = useState(false);
  const { token } = avanpostCookies.getTokens();

  const downloadFile = async () => {
    setLoading(true);

    const pendingToast = getPendingToast({
      icon: IconNames.DOWNLOAD,
      intent: "primary",
    });

    const showPendingToast = AppToaster.show(pendingToast);
    let successToastKey, errorToastKey;

    try {
      const response = await ky
        .get(`${BACKEND_URL}/rest/processings/${url}`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        .blob(); // Get the response as a Blob

      if (response.size > 0) {
        const downloadLink = window.URL.createObjectURL(response);

        const link = document.createElement("a");
        link.href = downloadLink;
        link.download = `features.geojson`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        const successToast = getSuccessToast(
          () => t`Successfully prepared export data`,
          response,
        )("success");

        successToastKey = AppToaster.show(successToast);
      } else {
        const errorToast = getErrorToast(
          () => t`The selected AOIs contain no data`,
          response,
        )("danger");

        errorToastKey = AppToaster.show(errorToast);
      }
    } catch (error) {
      const errorToast = getErrorToast(
        () => t`Error exporting processing result`,
        error,
      )("danger");

      errorToastKey = AppToaster.show(errorToast);
    } finally {
      AppToaster.dismiss(showPendingToast);

      if (successToastKey) {
        setTimeout(() => AppToaster.dismiss(successToastKey), 5000);
      }

      if (errorToastKey) {
        setTimeout(() => AppToaster.dismiss(errorToastKey), 5000);
      }

      setLoading(false);
    }
  };

  return { downloadFile, loading };
};