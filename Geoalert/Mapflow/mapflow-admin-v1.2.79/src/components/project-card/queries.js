import { gql } from "@apollo/client";
import client from "graphql/client";

const DELETE_PROJECT = gql`
  mutation deleteProject($id: ID!) {
    deleteProject(id: $id)
  }
`;

export const deleteProject = async (id) => {
  const result = await client.mutate({
    mutation: DELETE_PROJECT,
    fetchPolicy: "no-cache",
    variables: { id },
  });
  return result?.data?.deleteProject;
};
