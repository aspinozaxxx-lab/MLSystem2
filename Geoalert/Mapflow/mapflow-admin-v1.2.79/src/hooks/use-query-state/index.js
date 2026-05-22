import { useCallback, useMemo } from "react";
import { useHistory, useLocation } from "react-router-dom";

import qs from "qs";
import { useLayoutEffect } from "react";

/**
 *
 * @param {string} query
 * @param {boolean} options.asInt
 * @param {(value: any) => void} options.onUpdate
 * @param {number|string|boolean} options.defaultValue
 * @returns {[string, setQuery]}
 */
export const useQueryState = (query, options) => {
  const location = useLocation();
  const history = useHistory();

  const queries = useMemo(
    () =>
      qs.parse(location.search, {
        ignoreQueryPrefix: true,
      }),
    [location.search],
  );

  const setQuery = useCallback(
    /**
     * @function setQuery
     *  @param {number|string|boolean} value
     * @param {Object} options
     * @param {boolean} options.silent
     */
    (value) => {
      const compQueries = { ...queries, [query]: value };
      const queryString = qs.stringify(compQueries, {
        skipNulls: true,
      });

      history.push(`${location.pathname}?${queryString}`);
      if (options.onUpdate) options.onUpdate(value);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [history, location, query, queries, options.onUpdate],
  );

  const searchValue = queries[query];
  const value =
    (options.asInt ? parseInt(searchValue) : searchValue) ||
    options.defaultValue;

  // update location on mount component
  useLayoutEffect(() => {
    // for update location with actual location and prevent loop
    const searchValue = queries[query];

    // eslint-disable-next-line eqeqeq
    if (searchValue == value) {
      return;
    }

    setQuery(value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [setQuery]);

  return [value, setQuery];
};
