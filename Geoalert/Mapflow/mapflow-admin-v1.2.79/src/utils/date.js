import { avanpostCookies } from "providers/auth-avanpost-provider";
import jwtDecode from "jwt-decode";

export const UTCRemove = (time) => time.substring(0, time.length - 5);

export const currentExpiresTime = () => {
  let time = 0;
  const token = 
  try {
    const decodedToken = 

    const expiresInMilliseconds = decodedToken.exp * 1000;
    const currentTime = Date.now();
    const timeRemaining = expiresInMilliseconds - currentTime;

    time = timeRemaining;
  } catch (error) {
    console.error("Error decoding token:, error);
  }

  return time > 0 ? time / 1000 : 1700;
};
