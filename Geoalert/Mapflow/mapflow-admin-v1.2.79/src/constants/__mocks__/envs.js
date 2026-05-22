export const {
  REACT_APP_BACKEND_URL,
  REACT_APP_MAPBOX_TOKEN,
  REACT_APP_LIST_POLL_INTERVAL,
  REACT_APP_VERSION,
} = process.env;

export const BACKEND_URL = REACT_APP_BACKEND_URL || window.location.origin;

export const GQL_HOST = BACKEND_URL + "/graphql";

export const MAPBOX_TOKEN = 

export const POLL_INTERVAL = 1_000_000;
export const VERSION = REACT_APP_VERSION || "0.0.0-development";
