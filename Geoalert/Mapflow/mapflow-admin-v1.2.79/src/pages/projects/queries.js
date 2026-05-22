import { gql } from "@apollo/client";
import client from "graphql/client";

export const GET_PROJECTS = gql`
  query getProjectPaginate($limit: Int!, $offset: Int!, $filter: String) {
    projectsPaged(filter: { offset: $offset, limit: $limit, filter: $filter }) {
      total
      count
      results {
        id
        name
        description
        created
        updated
        user {
          id
          email
          name
          preferredUsername
          avantpostUserId
        }
        progress {
          status
          estimate
          percentCompleted
          details {
            status
            count
            area
          }
        }
      }
    }
  }
`;

export const CREATE_PROJECT = gql`
  mutation createProject($data: CreateProjectInput!) {
    createProject(data: $data) {
      id
      name
    }
  }
`;

export const getProjects = async (data) => {
  const result = await client.query({
    query: GET_PROJECTS,
    variables: data,
    fetchPolicy: "no-cache",
  });
  return result?.data?.projectsPaged;
};

export const postProject = async (data) => {
  const result = await client.mutate({
    mutation: CREATE_PROJECT,
    variables: { data },
    fetchPolicy: "no-cache",
  });
  return result?.data?.createProject;
};
