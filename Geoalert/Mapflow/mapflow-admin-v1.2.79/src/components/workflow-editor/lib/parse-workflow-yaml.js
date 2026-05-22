import { decode } from "base-64";
import { load, dump } from "js-yaml";

const safeDecode = (base64) => {
  try {
    return decode(base64);
  } catch {
    return `# Base64 decoding error, while decoding value: ${base64}`;
  }
};

/**
 *
 * @param {String} str
 */

export const parseWorkflowYaml = (str) => {
  const base64Fields = {};
  const object = load(str);

  const loop = (v) => {
    if (typeof v === "object" && v !== null) {
      if (Array.isArray(v)) return;
      for (const [key, value] of Object.entries(v)) {
        if (key.startsWith("requirements") || key.startsWith("pipeline")) {
          const newValue = typeof value === "string" ? safeDecode(value) : null;
          base64Fields[key] = newValue;
          v[key] = `$FILE_${key}`;
        } else if (typeof value === "object") {
          loop(value);
        }
      }
    }
  };

  loop(object);

  return {
    workflow: dump(object),
    ...base64Fields,
  };
};
