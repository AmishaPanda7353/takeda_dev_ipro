import React from "react";
import ReactDOM from "react-dom/client";
import { Provider } from "react-redux";
import { PublicClientApplication } from '@azure/msal-browser'
import { MsalProvider } from '@azure/msal-react'
import App from "./App";
import "./styles/index.css";
import reportWebVitals from "./reportWebVitals";
import "./assets/fonts/Tasty-Gothic_Demi.ttf";
import store from "./store";
import { msalConfig } from './auth.config';

const msalInstance = new PublicClientApplication(msalConfig)

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);
root.render(
  <React.StrictMode>
    <MsalProvider instance={msalInstance}>
      <Provider store={store}>
        <App />
      </Provider>
    </MsalProvider>
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
