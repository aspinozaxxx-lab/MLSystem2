/* istanbul ignore file */
import React from "react";
import ReactDOM from "react-dom";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { ApolloProvider } from "@apollo/client";
import * as serviceWorker from "./serviceWorker";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import "styles/styles.scss";
import App from "app";
import client from "graphql/client";
import { LanguageProvider, ThemeProvider } from "providers";
import { VERSION, isProduction } from "constants/envs";
import { AuthProvider } from "providers/auth-avanpost-provider/AuthProvider";

console.log(
  `%cAPP VERSION ${VERSION}.`,
  "color: #137CBD; font-weight: bold; font-style: italic; background-color: #FFFFFF; padding: 2px;",
);

const queryClient = new QueryClient();

ReactDOM.render(
  <QueryClientProvider client={queryClient}>
    <ApolloProvider client={client}>
      <LanguageProvider>
        <ThemeProvider>
          <div className="app">
            <AuthProvider>
              <App />
            </AuthProvider>
          </div>
        </ThemeProvider>
      </LanguageProvider>
    </ApolloProvider>
    {!isProduction && <ReactQueryDevtools position="bottom-left" />}
  </QueryClientProvider>,
  document.getElementById("root"),
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
