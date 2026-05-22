import { useApolloClient } from "@apollo/client";
import { t } from "@lingui/macro";
import { useQuery } from "@tanstack/react-query";
import { GET_PROJECT } from "pages/link-workflow";
import { getErrorToast, showToast } from "toaster";

const useProjectQuery = (projectId) => {
  const client = useApolloClient();

  return useQuery({
    queryKey: ["getProject", projectId],
    queryFn: async () => {
      const projectInfo = await client.query({
        query: GET_PROJECT,
        fetchPolicy: "no-cache",
        variables: { projectId },
      });
      return projectInfo?.data?.project;
    },
    onError: (e) => {
      showToast(getErrorToast(t`Error fetching project info`));
    },
  });
};

export default useProjectQuery;
