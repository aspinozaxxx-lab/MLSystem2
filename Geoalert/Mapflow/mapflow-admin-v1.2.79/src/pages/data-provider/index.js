import { useQuery } from "@tanstack/react-query";
import client from "graphql/client";
import React, { useMemo } from "react";
import { GET_DATA_PROVIDERS } from "../../components/data-provider/queries";

import { POLL_INTERVAL } from "constants/envs";
import { Trans } from "@lingui/react";
import DataProviderContent from "components/data-provider/data-provider-content";

import { Tag } from "@blueprintjs/core";
import DataProviderHeader from "components/data-provider/data-provider-header";
import useDebounce from "hooks/use-debounce";

import DataProviderActions from "../../components/data-provider/actions";

import * as routes from "constants/routes";

import { useHistory } from "react-router-dom";

const DataProvider = () => {
  const history = useHistory();
  const [dataProviderSearch, setDataProviderSearch] = React.useState("");

  const debouncedSearch = useDebounce(dataProviderSearch, 500);

  const { data, status } = useQuery({
    queryKey: ["dataProviders"],
    queryFn: async () => {
      const result = await client.query({
        query: GET_DATA_PROVIDERS,
        fetchPolicy: "no-cache",
      });
      return result.data.dataProviders;
    },
    refetchInterval: POLL_INTERVAL,
    refetchOnWindowFocus: false,
    keepPreviousData: true,
  });

  const dataProviders = useMemo(() => {
    const wds = [...(data || [])];
    wds.sort((a, b) => {
      if (a.isDefault === b.isDefault) {
        return 0;
      }
      if (a.isDefault) {
        return -1;
      }
      return 1;
    });

    return wds.filter(({ displayName, name  }) => {
      const searchedValue = displayName? displayName: name
      return searchedValue.toLowerCase().includes(debouncedSearch.trim().toLowerCase())
    }
    );
  }, [data, debouncedSearch]);

  const columns = useMemo(
    () => [
      {
        Header: <Trans id="Name" />,
        id: "displayName",
        accessor: "displayName",
        Cell: ({ row }) => {
          const { name, displayName, isDefault } = row.original;
          return (
            <div className="workflow-cell">
              <span className="workflow-cell__name">{displayName || name}</span>
              {isDefault && (
                <Tag round>
                  <Trans id="default" />
                </Tag>
              )}
            </div>
          );
        },
      },
      {
        Header: <Trans id="Actions" />,
        Cell: ({ row }) => {
          const { id, name, displayName, isDefault } = row.original;
          return (
            <DataProviderActions
              dataProviderId={id}
              isDefault={isDefault}
              name={displayName || name}
            />
          );
        },
        id: "actions",
        disableSortBy: false,
      },
    ],
    [],
  );

  return (
    <div className="projects">
      <DataProviderHeader
        onCreate={() => history.push(routes.DATA_PROVIDER_CREATE)}
        setDataProviderSearch={setDataProviderSearch}
        dataProviderSearch={dataProviderSearch}
      />

      <DataProviderContent
        status={status}
        data={dataProviders}
        columns={columns}
      />
    </div>
  );
};

export default DataProvider;
