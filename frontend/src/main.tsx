import React from "react";
import ReactDOM from "react-dom/client";

import { App } from "./App";
import { installPseudoModuleRecovery } from "./utils/moduleRecovery";
import "./styles/index.css";

installPseudoModuleRecovery();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
