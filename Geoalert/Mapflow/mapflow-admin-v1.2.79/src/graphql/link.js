import { t } from "@lingui/macro";
import { createUploadLink } from "apollo-upload-client";
import { onError } from "@apollo/client/link/error";
import { split } from "@apollo/client";
import { BatchHttpLink } from "@apollo/client/link/batch-http";

import { showToastT } from "toaster";
import { GQL_HOST } from "constants/envs";
import { ToastTemplates } from "constants/toasts";
import client from "./client";

import { setContext } from "@apollo/client/link/context";
import { avanpostCookies } from "providers/auth-avanpost-provider";

const authLink = setContext(({ headers = {} }) => {
  const { token } = avanpostCookies.getTokens();

  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : "",
    },
  };
});

const signOut = async () => {
  localStorage.removeItem("token");
};

// TODO add tests for it
const signOutWithToast = async (message) => {
  await signOut();
  try {
    await client.clearStore();
  } catch (error) {
    console.error(error);
  } finally {
    showToastT({ message, ...ToastTemplates.ERROR });
  }
};

const errorLink = onError(({ graphQLErrors, operation, forward }) => {
  if (graphQLErrors) {
    for (let e of graphQLErrors) {
      switch (e.code) {
        case "AUTHENTICATION_ERROR": // process it on the sign-in form
          return forward(operation);
        case "ACCESS_DENIED":
          signOutWithToast(t`Access denied`);
          break;
        case "BAD_TOKEN":
          signOutWithToast(t`Bad token`);
          break;
        case "TOKEN_EXPIRED":
          signOutWithToast(t`Token expired`);
          break;
        default:
          console.error("Uncaught error", e);
          return forward(operation);
      }
    }
  }
});

// const LINK_OPTS = { uri: GQL_HOST, credentials: "include" };
const LINK_OPTS = {
  uri: (operation) =>
    `${GQL_HOST}?operation=${encodeURIComponent(operation.operationName)}`,
  credentials: "include",
};
const httpLink = createUploadLink(LINK_OPTS);

export const batchLink = split(
  (operation) => operation.getContext().hasUpload,
  createUploadLink(LINK_OPTS),
  new BatchHttpLink(LINK_OPTS),
);

export default errorLink.concat(authLink.concat(httpLink));
