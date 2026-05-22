import { client } from ".";

export const postRasters = async (formData) => {
  return client.post("rasters", { body: formData, timeout: 6000_00 }).json();
};

export const getRasters = async () => {
  return client.get("rasters/mosaic").json();
};
