import { gql } from "@apollo/client";

export const GET_DATA_PROVIDERS = gql`
  query getDataProviders {
    dataProviders {
      id
      name
      displayName
      urlTemplate
      previewUrl
      credentialsToken
      credentialsUsername
      credentialsPassword
      isDefault
      mapfileUri
    }
  }
`;

export const GET_DATA_PROVIDER = gql`
  query getDataProvider($id: [ID!]) {
    dataProviders(ids: $id) {
      id
      name
      displayName
      urlTemplate
      previewUrl
      credentialsUsername
      credentialsPassword
      credentialsToken
      isDefault
      mapfileUri
    }
  }
`;

// create data provider
export const CREATE_DATA_PROVIDER = gql`
  mutation createDataProvider($data: CreateDataProviderInput!) {
    createDataProvider(data: $data) {
      id
      name
      displayName
      urlTemplate
      previewUrl
      credentialsUsername
      credentialsPassword
      credentialsToken
      isDefault
      mapfileUri
    }
  }
`;

//  update data provider
export const UPDATE_DATA_PROVIDER = gql`
  mutation updateDataProvider($data: UpdateDataProviderInput!) {
    updateDataProvider(data: $data) {
      id
      name
      displayName
      urlTemplate
      previewUrl
      credentialsUsername
      credentialsPassword
      credentialsToken
      isDefault
      mapfileUri
    }
  }
`;

// delete
export const DELETE_DATA_PROVIDER = gql`
  mutation deleteDataProvider($id: ID!) {
    deleteDataProvider(id: $id)
  }
`;

//link unlink
export const LINK_DATA_PROVIDER = gql`
  mutation linkDataProvider($userId: ID!, $dataProviderId: ID!) {
    linkDataProvider(userId: $userId, dataProviderId: $dataProviderId)
  }
`;

export const UNLINK_DATA_PROVIDER = gql`
  mutation unlinkDataProvider($userId: ID!, $dataProviderId: ID!) {
    unlinkDataProvider(userId: $userId, dataProviderId: $dataProviderId)
  }
`;

// users
export const GET_USERS_LINKED_TO_DATA_PROVIDER = gql`
  query getDataProviderUsers($id: ID!) {
    dataProviderUsers(id: $id) {
      id
      email
    }
  }
`;
