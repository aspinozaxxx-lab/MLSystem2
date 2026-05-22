import { useAuth } from "providers/auth-avanpost-provider/AuthProvider";

const UserRoles = {
  ADMIN: "ADMIN",
  USER: "USER",
};

export const useUser = () => {
  const { isAuth, user } = useAuth();

  return { ...user, isAuth };
};

// useUser().role === UserRoles.ADMIN
export const useIsAdmin = () => useUser().isAdmin;

export const AdminRender = ({ children }) => {
  const isAdmin = useIsAdmin();

  if (isAdmin) {
    return <>{children}</>;
  }

  return null;
};
