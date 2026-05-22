import React from "react";
import { Trans, t } from "@lingui/macro";
import { useParams, generatePath } from "react-router-dom";
import { Button, Intent, H5 } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";
import ConfirmDialog from "components/confirm-dialog";
import { setTestId } from "test-utils/set-testid";
import { showToast, getSuccessToast, getErrorToast } from "toaster";
import * as routes from "constants/routes";

export const DELETE_PROCESSING = gql`
  mutation deleteProcessing($id: ID!) {
    deleteProcessing(id: $id)
  }
`;

export const deleteProcessing = (client) => async (id) => {
  const result = await client.mutate({
    mutation: DELETE_PROCESSING,
    fetchPolicy: "no-cache",
    variables: { id },
  });
  return result?.data?.deleteProcessing;
};

function Actions({ id, name }) {
  const { projectId } = useParams();

  const client = useApolloClient();
  const queryClient = useQueryClient();
  const mutation = useMutation(deleteProcessing(client), {
    onSuccess: () => {
      queryClient.invalidateQueries(["processings", projectId]);
      showToast(
        getSuccessToast(t`Processing "${name}" deleted`, {
          icon: IconNames.TRASH,
        }),
      );
    },
    onError: () => showToast(getErrorToast(t`Error delete processing`)),
  });

  // const handleNewTabClick = (e) => {
  //   e.stopPropagation();
  //   window.open(
  //     generatePath(routes.PROJECT_PROCESSINGS, { projectId: id }),
  //     "_blank",
  //   );
  // };

  return (
    <div className="processing_actions">
      {/* <Button
        minimal
        icon={IconNames.SHARE}
        intent={Intent.PRIMARY}
        onClick={handleNewTabClick}
      /> */}

      <ConfirmDialog
        className="processing_actions__remove-btn"
        intent={Intent.DANGER}
        icon={IconNames.TRASH}
        confirmButtonText={<Trans id="Delete" />}
        cancelButtonText={<Trans id="Cancel" />}
        text={
          <H5>
            <Trans id="Confirm delete processing" />
          </H5>
        }
        onConfirm={(close) => {
          mutation.mutate(id);
          close();
        }}
      >
        {({ showDialog }) => (
          <Button
            minimal
            elementRef={setTestId`delete-processing`}
            intent={Intent.DANGER}
            icon={IconNames.TRASH}
            loading={mutation.isLoading}
            onClick={showDialog}
          />
        )}
      </ConfirmDialog>
    </div>
  );
}

export default React.memo(Actions);
