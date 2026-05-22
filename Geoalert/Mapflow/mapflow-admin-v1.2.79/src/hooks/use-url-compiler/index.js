import { compile } from "path-to-regexp";

function useUrlCompiler() {
  return (path, args) => compile(path, { encode: encodeURIComponent })(args);
}

export { useUrlCompiler };
