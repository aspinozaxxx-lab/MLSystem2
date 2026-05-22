import ky from "ky";

import { BACKEND_URL } from "constants/envs";
import { avanpostCookies } from "providers/auth-avanpost-provider";

/**
 *
 * @param {Request} request
 */
function attachBearerToken(request) {
  const { token } = avanpostCookies.getTokens();
  if (token) {
    request.headers.set("authorization", `Bearer ${token}`);
  }
}

export const client = ky.extend({
  prefixUrl: BACKEND_URL + "/rest",
  hooks: {
    beforeRequest: [attachBearerToken],
  },
});
