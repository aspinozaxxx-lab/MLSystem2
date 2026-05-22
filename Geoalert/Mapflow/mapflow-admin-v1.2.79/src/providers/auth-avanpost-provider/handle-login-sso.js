import { authorizationCode } from "shared/api/client/avanpost";
import { lcKeyAuthBackUrl, lcKeyCodeVerifier, lcKeyState } from "./model";

const clearCodeStates = () => {
  localStorage.removeItem(lcKeyAuthBackUrl);
  localStorage.removeItem(lcKeyState);
  localStorage.removeItem(lcKeyCodeVerifier);
};

/**
 * Действия после редиректа авторизации
 * TODO вынести вызов выше и через await
 */

export const handleLoginSSO = async (onSuccess) => {
  return new Promise((resolve, reject) => {
    if (typeof window !== "undefined") {
      const { searchParams } = new URL(window.location.toString());

      const code = searchParams.get("code");
      const state = searchParams.get("state");
      const error = searchParams.get("error");

      if (error) reject();

      const backUrl = localStorage.getItem(lcKeyAuthBackUrl);
      const localState = localStorage.getItem(lcKeyState);
      const codeVerifier = localStorage.getItem(lcKeyCodeVerifier);

      // Подчищаем backUrl после редиректа
      if (backUrl) localStorage.removeItem(lcKeyAuthBackUrl);

      // Сохранить токены по коду из урла и защита от CSRF
      if (code && codeVerifier && state === localState) {
        authorizationCode(code, codeVerifier)
          .then((response) => {
            // Response - expires_in, access_token, refresh_token
            onSuccess(response);
            clearCodeStates();
            resolve();
            window.location.href = backUrl || window.location.origin;
          })
          .catch(reject);
      } else {
        // Continue login by SSO
        resolve();
      }
    } else {
      reject();
    }
  });
};
