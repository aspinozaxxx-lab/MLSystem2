import { gql } from "@apollo/client";
import client from "graphql/client";

export const CREATE_AOIS_FROM_FILE = gql`
  mutation createAoisFromFile(
    $processingId: ID!
    $file: Upload!
    $mergeStrategy: MergeStrategy
  ) {
    createAoisFromFile(
      data: {
        processingId: $processingId
        file: $file
        mergeStrategy: $mergeStrategy
      }
    ) {
      count
      bbox
    }
  }
`;

export const createAoisFromFile = async (data) => {
  const result = await client.mutate({
    mutation: CREATE_AOIS_FROM_FILE,
    variables: { data },
    fetchPolicy: "no-cache",
  });
  return result?.data?.createAoisFromFile;
};
