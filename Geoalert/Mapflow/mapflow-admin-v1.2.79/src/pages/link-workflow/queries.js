import { useQuery } from "@tanstack/react-query";
import { useApolloClient } from "@apollo/client";
import { GET_PROJECT_WORKFLOWS } from "components/project-workflow-list";

export const useLinkedProjectWorkflowsQuery = (projectId) => {
  const client = useApolloClient();

  return useQuery({
    queryKey: ["linkedWorkflows", projectId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_PROJECT_WORKFLOWS,
        fetchPolicy: "no-cache",
        variables: { projectId },
      });

      return result?.data?.project?.workflowDefs;
    },
  });
};
