import { PROJECTS } from "constants/routes";
import { useIsAdmin } from "providers/auth/use-user";
import { Redirect, Route } from "react-router-dom";

export const PrivateRoute = (props) => {
  const isAdmin = useIsAdmin();

  if (!isAdmin) {
    return <Redirect to={PROJECTS} />;
  }

  return <Route {...props} />;
};
