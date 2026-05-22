import { useMemo } from "react";
import { matchPath, useLocation } from "react-router-dom";

import { ALL_ROUTES } from "constants/routes";

export const getActivePath = (location) => {
  return ALL_ROUTES.find((path) => matchPath(location, { path })?.isExact);
};

export const useActiveTab = () => {
  const location = useLocation();

  return useMemo(() => getActivePath(location.pathname), [location.pathname]);
};

export const useGetRouteParam = (key) => {
  const location = useLocation();

  return useMemo(() => {
    for (let path of ALL_ROUTES) {
      const match = matchPath(location.pathname, { path });
      const value = match?.params?.[key];
      if (value) return value;
    }

    return null;
  }, [location.pathname, key]);
};
