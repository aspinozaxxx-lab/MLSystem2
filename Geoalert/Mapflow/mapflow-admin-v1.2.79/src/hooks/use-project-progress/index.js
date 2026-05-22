import { gql, useApolloClient } from "@apollo/client";
import { useQuery } from "@tanstack/react-query";
import { POLL_INTERVAL } from "constants/envs";

const GET_PROJECT = gql`
  query getProject($ids: [ID!]) {
    projects(ids: $ids) {
      progress {
        status
        estimate
        percentCompleted
        details {
          count
          status
          area
          statusUpdateDate
        }
      }
    }
  }
`;

export const useProjectProgress = (projectId) => {
  const client = useApolloClient();

  const {
    data: projectProgressResult,
    loading: projectProgressLoading,
  } = useQuery({
    queryKey: ["getProjectProgress", projectId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_PROJECT,
        fetchPolicy: "no-cache",
        variables: { ids: [projectId] },
      });
      return result?.data.projects[0].progress;
    },
    refetchInterval: POLL_INTERVAL,
  });

  return { projectProgressResult, projectProgressLoading };
};
