import { createAuthCookiesControl } from "shared/cookies/create-auth";
import { generateAuthFormUrl } from "shared/api/client/avanpost";

import { generateRandomString } from "./lib";

export const avanpostCookies = createAuthCookiesControl(
  "AVANPOST_TOKEN",
  "AVANPOST_REFRESH_TOKEN",
);

export const lcKeyState = "authStateKey";
export const lcKeyCodeVerifier = "authCodeVerifier";
export const lcKeyAuthBackUrl = "authBackUrl";
const ACTION_LOGIN_SSO = "login-sso";
export const ACTION_SUCCES_AUTH = "succes-auth";
const ACTION_LOGOUT = "logout-sso";
const ACTION_START_LOADING = "start-loading";
const ACTION_AUTH_FAILED = "auth-failed";

export const initialState = {
  user: null,
  isAuth: false,
  isLoading: false,
  isAuthError: false,
};

export const AuthReducer = (state, action) => {
  switch (action.type) {
    case ACTION_START_LOADING: {
      return { ...state, isLoading: true };
    }
    case ACTION_SUCCES_AUTH: {
      return {
        ...state,
        isAuth: true,
        isLoading: false,
        isAuthError: false,
        user: action.payload,
      };
    }
    case ACTION_LOGIN_SSO: {
      const codeVerifier = generateRandomString();
      const state = generateRandomString();

      localStorage.setItem(lcKeyState, state);
      localStorage.setItem(lcKeyCodeVerifier, codeVerifier);
      generateAuthFormUrl(codeVerifier, state).then((url) => {
        localStorage.setItem(lcKeyAuthBackUrl, window.location.href);
        window.location.replace(url);
      });

      return { ...state, isLoading: true };
    }
    case ACTION_LOGOUT: {
      return {
        ...initialState,
      };
    }
    case ACTION_AUTH_FAILED: {
      return { ...initialState, isAuthError: true };
    }
    default:
      return state;
  }
};

const createAction = (type, payload) => ({ type, payload });

export const AUTH_ACTIONS = {
  ACTION_LOGIN_SSO: createAction(ACTION_LOGIN_SSO),
  ACTION_LOGOUT: createAction(ACTION_LOGOUT),
  ACTION_SUCCES_AUTH: createAction(ACTION_SUCCES_AUTH),
  ACTION_START_LOADING: createAction(ACTION_START_LOADING),
  ACTION_AUTH_FAILED: createAction(ACTION_AUTH_FAILED),
};
