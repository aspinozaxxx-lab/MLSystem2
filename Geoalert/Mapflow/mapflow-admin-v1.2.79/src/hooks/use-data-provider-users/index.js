import { useApolloClient } from "@apollo/client";
import { getErrorToast, showToast } from "toaster";
import { useQuery } from "@tanstack/react-query";
import { GET_USERS_LINKED_TO_DATA_PROVIDER } from "components/data-provider/queries";
import { t } from "@lingui/macro";

function useDataProviderUsers(dataProviderId) {
  const client = useApolloClient();

  const {
    data: users,
    isLoading: usersLoading,
    refetch: refetchUsers,
  } = useQuery({
    queryKey: ["dataProviderUsers", dataProviderId],
    queryFn: async () => {
      const result = await client.query({
        query: GET_USERS_LINKED_TO_DATA_PROVIDER,
        fetchPolicy: "no-cache",
        variables: { id: dataProviderId },
      });
      return result?.data?.dataProviderUsers;
    },
    onError: () => {
      showToast(getErrorToast(t`Failed to get users linked to data provider`));
    },
  });

  return { users, usersLoading, refetchUsers };
}

export default useDataProviderUsers;
