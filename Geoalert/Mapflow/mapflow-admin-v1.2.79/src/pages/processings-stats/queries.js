import { gql } from "@apollo/client";
import client from "graphql/client";

export const GET_PROCESSINGS_PAGED = gql`
  query getProcessingsPaged(
    $limit: Int!
    $offset: Int!
    $terms: String
    $dateFrom: Date
    $dateTo: Date
    $sortOrder: ProcessingSortOrder
    $sortBy: SortBy
    $statuses: [Status!]
  ) {
    processingsPaged(
      filters: {
        offset: $offset
        limit: $limit
        terms: $terms
        dateFrom: $dateFrom
        dateTo: $dateTo
        sortOrder: $sortOrder
        sortBy: $sortBy
        statuses: $statuses
      }
    ) {
      count
      total
      results {
        id
        projectId
        name
        projectName
        email
        created
        updated
        description
        area
        user {
          id
          name
          email
          preferredUsername
        }
        progress {
          status
          percentCompleted
          completionDate
          estimate
        }
        messages {
          message
        }
        workflowDef {
          name
        }
      }
    }
  }
`;

export const getProcessingsPaged = async (data) => {
  const result = await client.query({
    query: GET_PROCESSINGS_PAGED,
    variables: data,
    fetchPolicy: "no-cache",
  });
  return result?.data?.processingsPaged;
};
