import { client } from ".";

export const getUserStatus = () => {
  return client.get("user/status", { retry: 0 }).json();
};
