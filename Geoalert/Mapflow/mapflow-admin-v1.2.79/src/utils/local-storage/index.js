/**
 *
 * @param {string} key
 * @param {any} value
 */
 const set = (key, value) => {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.error(error);
  }
};

/**
 *
 * @param {string} key
 */
const get = (key, fallback) => {
  try {
    const value = window.localStorage.getItem(key);
    if (typeof value !== "string") return fallback;
    return JSON.parse(value);
  } catch (error) {
    console.error(error);
    return fallback;
  }
};

const LocalStorage = { set, get };

export default LocalStorage;
