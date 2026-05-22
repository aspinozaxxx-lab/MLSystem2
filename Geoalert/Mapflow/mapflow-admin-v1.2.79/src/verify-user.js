import decode from "jwt-decode";

function getUserInfo(token) {
  if (!token) {
    return {};
  }

  try {
    return decode(token);
  } catch (error) {
    console.error("Unable to decode user token", error);
    throw error;
  } 
}

export function verifyUser(token) {
  const claim = getUserInfo(token)
  const userRole = claim.resource_access?.whitemaps?.roles?.map(role => role.toUpperCase())
    .includes("ADMIN") ? "ADMIN" : "USER";
  const isLoggedIn = Boolean(token && userRole);
  return { isLoggedIn, userRole };
}
