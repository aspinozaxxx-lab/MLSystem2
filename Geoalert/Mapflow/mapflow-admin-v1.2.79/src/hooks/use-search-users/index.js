import { gql, useApolloClient } from '@apollo/client';
import { useQuery } from '@tanstack/react-query';


const GET_USERS = gql`
  query getUsers($ids: [ID!], $emails: [String!], $roles: [Role!]) {
    users(ids: $ids, emails: $emails, roles: $roles) {
      id
      email
    }
  }
`;

// Custom hook for searching users
export  const useSearchUser = (debouncedSearchInput) => {
  const client = useApolloClient();

  const { data: searchedUser, isLoading: searchedUserLoading } = useQuery({
    queryKey: ["user", debouncedSearchInput],
    queryFn: async () => {
      const result = await client.query({
        query: GET_USERS,
        fetchPolicy: "no-cache",
        variables: { emails: [debouncedSearchInput] },
      });
      return result?.data?.users[0] || null;
    },
    enabled: !!debouncedSearchInput,
  });

  return { searchedUser, searchedUserLoading };
};