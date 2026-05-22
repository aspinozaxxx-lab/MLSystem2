function parseNumber(value, fallback = 0) {
  const maybeNumber = Number(value);

  if(!isNaN(maybeNumber)) {
    return maybeNumber
  }

  return fallback
}

export const isProduction = process.env.NODE_ENV === "production";

export const VERSION = process.env.REACT_APP_VERSION || "0.0.0-development";

export const BACKEND_URL = window.location.origin.includes('localhost') ? 'https://mapflow.dev.nspd.rosreestr.gov.ru' : process.env.REACT_APP_BACKEND_URL || window.location.origin;
export const GQL_HOST = BACKEND_URL + "/graphql";

export const MAPBOX_TOKEN = 
// "REDACTED_MAPBOX_TOKEN"

export const SSO_AVANPOST_URL = process.env.REACT_APP_SSO_AVANPOST_URL;
export const AVANPOST_CLIENT_ID = process.env.REACT_APP_AVANPOST_CLIENT_ID;

export const POLL_INTERVAL = parseNumber(process.env.REACT_APP_LIST_POLL_INTERVAL, 3_000);

export const AOI_VALIDATION = {
  MIN_AREA: parseNumber(process.env.REACT_APP_MIN_AREA, 0),
  MAX_AREA: parseNumber(process.env.REACT_APP_MAX_AREA, 100),
  MAX_SIZE: parseNumber(process.env.REACT_APP_MAX_SIZE, 120),
  MAX_FILE_SIZE_MB: parseNumber(process.env.REACT_APP_MAX_FILE_SIZE_MB, 512),
  MAX_IMAGE_SIZE_PIXELS: parseNumber(process.env.REACT_APP_MAX_IMAGE_SIZE_PIXELS, 30_000)
}

export const BASEMAP_URL = process.env.REACT_APP_BASE_MAP_URL;
