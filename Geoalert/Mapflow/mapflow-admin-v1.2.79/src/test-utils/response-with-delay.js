export const responseWithDelay = (
  result,
  { error = false, delay = 10 } = {},
) => () => {
  return new Promise((resolve, reject) => {
    if (error) reject(new Error(error));
    setTimeout(() => resolve(result), delay);
  });
};
