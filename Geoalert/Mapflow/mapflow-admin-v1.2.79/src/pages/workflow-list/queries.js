import { gql, useApolloClient } from "@apollo/client";
import { Intent } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";
import { t } from "@lingui/macro";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ErrorCodes } from "constants/common";
import { GET_USERS_LINKED_TO_WORKFLOW } from "pages/manage-workflow-users";
import { getErrorToast, showToast } from "toaster";

export const useUsersLinkedToWorkflowQuery = (workflowDefId) => {
  const client = useApolloClient();

  return useQuery({
    queryKey: ["workflowDefUsers", workflowDefId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_USERS_LINKED_TO_WORKFLOW,
        fetchPolicy: "no-cache",
        variables: { id: workflowDefId },
      });
      return result?.data?.workflowDefUsers;
    },
    onError: () => {
      showToast(getErrorToast(t`Failed to get users linked to workflow`));
    },
  });
};

export const DELETE_WORKFLOW = gql`
  mutation deleteWorkflowDef($id: ID!) {
    deleteWorkflowDef(id: $id)
  }
`;

export const useDeleteWorkflowMutation = (workflowDefId, workflowName) => {
  const client = useApolloClient();

  return useMutation({
    mutationKey: ["deleteWorkflow", workflowDefId],
    mutationFn: async ({ workflowDefId }) => {
      const result = await client.mutate({
        mutation: DELETE_WORKFLOW,
        variables: { id: workflowDefId },
      });
      return result?.data?.deleteWorkflowDef;
    },
    onSuccess: () => {
      showToast({
        message: t`Workflow "${workflowName}" deleted`,
        icon: IconNames.TRASH,
        intent: Intent.WARNING,
      });
    },
    onError: (error) => {
      const { graphQLErrors } = error;
      if (graphQLErrors)
        if (graphQLErrors.some(({ code }) => code === ErrorCodes.WD_IN_USE)) {
          showToast(
            getErrorToast(t`Workflow cannot be deleted while it is in use`),
          );
          return;
        }
      showToast(getErrorToast(t`Error delete workflow "${workflowName}"`));
    },
  });
};
