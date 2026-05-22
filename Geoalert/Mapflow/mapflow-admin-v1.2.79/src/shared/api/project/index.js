import { gql } from "@apollo/client";

export const UPDATE_PROJECT_NAME = gql`
  mutation updateProject($projectId: ID!, $name: String, $description: String) {
    updateProject(
      data: { id: $projectId, name: $name, description: $description }
    ) {
      id
      name
      description
    }
  }
`;

export const UPDATE_PROCESSING_NAME = gql`
  mutation updateProcessing($processingId: ID!, $name: String!) {
    updateProcessing(data: { processingId: $processingId, name: $name }) {
      id
      projectId
      name
    }
  }
`;
