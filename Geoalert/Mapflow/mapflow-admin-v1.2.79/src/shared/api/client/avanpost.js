import ky from "ky";

import { generateCodeChallenge } from "../../../providers/auth-avanpost-provider/lib";
import { AVANPOST_CLIENT_ID, SSO_AVANPOST_URL } from "constants/envs";

const requestedScopes = `profile,email,uid`;
const redirectUri = window.location.origin;

const clientId = window.location.origin.includes("localhost")
  ? "nspd"
  : AVANPOST_CLIENT_ID;
const ssoHost = SSO_AVANPOST_URL;

export async function generateAuthFormUrl(codeVerifier, state) {
  return Promise.resolve()
    .then(() => generateCodeChallenge(codeVerifier))
    .then((codeChallenge) => {
      const queryString = new URLSearchParams({
        response_type: "code",
        client_id: clientId,
        scope: requestedScopes,
        redirect_uri: redirectUri,
        code_challenge_method: "S256",
        code_challenge: codeChallenge,
        state,
      }).toString();

      return `${ssoHost}/oauth2/authorize?`.concat(queryString);
    });
}

export function endSession(token) {
  return ky.get(`${ssoHost}/oauth2/end_session`, {
    searchParams: [["id_token_hint", token]],
  });
}

function authToken(grantType, body) {
  const searchParams = new URLSearchParams({
    grant_type: grantType,
    client_id: clientId,
    ...body,
  });

  return ky
    .post(`${ssoHost}/oauth2/token`, {
      body: searchParams,
    })
    .json();
}

export function refreshAuthToken(refresh_token) {
  return authToken("refresh_token", { refresh_token });
}

export function authorizationCode(code, codeVerifier) {
  return authToken("authorization_code", {
    code,
    code_verifier: codeVerifier,
    redirect_uri: redirectUri,
  });
}
