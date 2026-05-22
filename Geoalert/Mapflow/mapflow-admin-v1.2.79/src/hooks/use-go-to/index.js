import { useHistory } from "react-router-dom";
import { useUrlCompiler } from "hooks/use-url-compiler";

export const useGoTo = (route, args = {}) => {
  const history = useHistory();
  const compile = useUrlCompiler();
  const goTo = (updateArgs = {}, updateRoute) => {
    const params = { ...args, ...updateArgs };
    history.push(compile(updateRoute || route, params));
  };
  return goTo;
};
