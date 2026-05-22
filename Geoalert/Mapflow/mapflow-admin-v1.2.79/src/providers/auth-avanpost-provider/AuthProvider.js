import { createContext, useContext, useEffect, useReducer } from "react";
import { Trans } from "@lingui/macro";
import { H2, Spinner } from "@blueprintjs/core";

import { handleLoginSSO } from "./handle-login-sso";
import {
  ACTION_SUCCES_AUTH,
  AUTH_ACTIONS,
  AuthReducer,
  avanpostCookies,
  initialState,
} from "./model";
import styles from "./styles.module.css";
import { AccessDenied } from "./ui/access-denied";

import { getUserStatus } from "shared/api/client/user";
import { refreshAuthToken } from "shared/api/client/avanpost";
import { currentExpiresTime } from "utils/date";

let timeout = null;

const AuthContext = createContext(initialState);
export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(AuthReducer, initialState);

  useEffect(() => {
    async function init() {
      try {
        dispatch(AUTH_ACTIONS.ACTION_START_LOADING);
        await handleLoginSSO((tokens) =>
          avanpostCookies.setTokens(
            tokens.access_token,
            tokens.refresh_token,
            tokens.expires_in,
          ),
        );

        // If dont has tokens login by sso and get it
        if (avanpostCookies.isTokensEmpty()) {
          dispatch(AUTH_ACTIONS.ACTION_LOGIN_SSO);
          return;
        }

        // If has tokens, fetch user/status for validating user, if invalid clear tokens
        const user = await getUserStatus();
        dispatch({ type: ACTION_SUCCES_AUTH, payload: user });

        const startRefreshTimeout = () => {
          const intervalFunction = async () => {
            try {
              const response = await refreshAuthToken(
                avanpostCookies.getTokens().refreshToken,
              );
              avanpostCookies.setTokens(
                response?.access_token,
                response?.refresh_token,
                response?.expires_in,
              );

              // Clear the existing timeout
              clearTimeout(timeout);

              // Set a new timeout for the next refresh
              timeout = setTimeout(
                intervalFunction,
                (response?.expires_in - 50) * 1000,
              );
            } catch (error) {
              // Handle refresh token error (e.g., token expired)
              console.error("Error refreshing token:, error);
            }
          };

          // Set the initial timeout
          timeout = setTimeout(
            intervalFunction,
            (currentExpiresTime() - 50) * 1000,
          );
        };

        // Start the initial refresh timeout
        startRefreshTimeout();
      } catch (error) {
        dispatch(AUTH_ACTIONS.ACTION_AUTH_FAILED);
        console.error(error, "error while autorizing");
        // avanpostCookies.clearSession();
      }
    }
    init();
  }, [state.isAuth]);

  const handleLogout = () => {
    try {
      dispatch(AUTH_ACTIONS.ACTION_LOGOUT);
      clearTimeout(timeout);
    } catch (error) {
      console.error(error);
    }
  };

  if (state.isLoading) {
    return (
      <div className={styles.pageLoader}>
        <Spinner style={{ margin: "1rem 0" }} />
        <H2>
          <Trans>Authentication...</Trans>
        </H2>
      </div>
    );
  }

  if (state.isAuthError) {
    return (
      <div className={styles.pageLoader}>
        <H2>
          <Trans>Authentication error</Trans>
        </H2>
      </div>
    );
  }

  if (state.isAuth) {
    return (
      <AuthContext.Provider
        value={{
          ...state,
          handleLogout,
        }}
      >
        {children}
      </AuthContext.Provider>
    );
  }

  return <AccessDenied />;
};
