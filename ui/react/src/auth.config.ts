import { Configuration, PopupRequest, EndSessionPopupRequest } from "@azure/msal-browser";

export const msalConfig: Configuration = {
  auth: {
    clientId: `${process.env.REACT_APP_CLIENT_ID}`,
    authority: `https://login.microsoftonline.com/${process.env.REACT_APP_MS_TENANTID}`, // This is a URL (e.g. https://login.microsoftonline.com/{your tenant ID})
    redirectUri: `${process.env.REACT_APP_REDIRECT_URL}`,
    postLogoutRedirectUri: `${process.env.REACT_APP_REDIRECT_URL}`,
  },
  cache: {
    cacheLocation: 'sessionStorage', // This configures where your cache will be stored
    storeAuthStateInCookie: false, // Set this to "true" if you are having issues on IE11 or Edge
  },
}

// Add scopes here for ID token to be used at Microsoft identity platform endpoints.
export const loginRequest: PopupRequest = {
  scopes: ['User.Read', 'openid', 'profile'],
}

export const logoutRequest: EndSessionPopupRequest = {
  mainWindowRedirectUri: `${process.env.REACT_APP_REDIRECT_URL}`,
}

// Add the endpoints here for Microsoft Graph API services you'd like to use.
export const graphConfig = {
  graphMeEndpoint: 'https://graph.microsoft.com/v1.0/me',
}
