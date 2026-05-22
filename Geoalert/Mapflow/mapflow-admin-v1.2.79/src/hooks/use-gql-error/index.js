const onError = (codes, defaultFn) => (e) => {
  for (let err of e.graphQLErrors) {
    const fn = err.code ? codes[err.code] : null;
    if (fn) return fn(err);
    return defaultFn(e);
  }
};

export function useGqlError(codes, defaultFn) {
  return onError(codes, defaultFn);
}
