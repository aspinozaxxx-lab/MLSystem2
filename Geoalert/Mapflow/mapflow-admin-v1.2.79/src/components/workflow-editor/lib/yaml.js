import { dump, load } from "js-yaml";

export const isEmptyYaml = (yml) => {
  if (typeof yml === "string") {
    if (yml.trim() === "") return true;
    return clearify(yml).startsWith("null");
  }
  return true;
};

export const clearify = (yml) => dump(load(yml));
