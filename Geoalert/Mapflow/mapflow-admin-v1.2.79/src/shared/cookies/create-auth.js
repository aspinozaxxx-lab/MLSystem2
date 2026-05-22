import Cookies from "js-cookie";
import { getIsLocalhost } from "shared/client";

const getExpire = (exp) => new Date(Date.now() + exp * 1000);

/**
 * @type Cookies.CookieAttributes
 */
const secureOptions = {
  path: "/",
  secure: !getIsLocalhost(),
};

export const createAuthCookiesControl = (keyToken, keyRefreshToken) => {
  function setTokens(token, refreshToken, expires) {
    if (!token || !refreshToken) {
      console.warn(
        `Token recieved "undefined" is ${!token && "access_token"} ${
          !refreshToken && "refresh_token"
        }`,
      );

      return;
    }

    const options = { ...secureOptions, expires: getExpire(expires) };

    Cookies.set(keyToken, token, options);
    Cookies.set(keyRefreshToken, refreshToken, secureOptions);
  }

  function getTokens() {
    return {
      token: ,
      refreshToken: ,
    };
  }

  function clearSession() {
    Cookies.remove(keyToken, secureOptions);
    Cookies.remove(keyRefreshToken, secureOptions);
  }

  return {
    setTokens,
    getTokens,
    clearSession,
    isTokensEmpty:  => {
      const { token, refreshToken } = getTokens();
      return !token || !refreshToken;
    },
  };
};
