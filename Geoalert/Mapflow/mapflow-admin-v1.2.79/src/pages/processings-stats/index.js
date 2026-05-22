import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { DateRangeInput2 } from "@blueprintjs/datetime2";

import { Button, Divider } from "@blueprintjs/core";

import useDebounce from "hooks/use-debounce";
import usePrevious from "hooks/use-previous";

import { getProcessingsPaged } from "./queries";

import ProcessingsHeader from "./header";
import { Pagination } from "components";
import { DATE_FNS_FORMATS } from "shared/date/date";
import {
  DEFAULT_SHORTCUTS,
  MONTHS,
  WEEKDAYS,
} from "shared/date/daterange-shortcuts";
import { IconNames } from "@blueprintjs/icons";
import { ProcessingsContent } from "./processings-content";
import { t } from "@lingui/macro";
import MultipleSelect from "components/multipleSelect";

const DEFAULT_PAGE_SIZE = 20;

function ProcessingsStats() {
  const [search, setSearch] = useState("");
  const [dateRange, setDateRange] = useState([null, null]);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [page, setPage] = useState(1);
  const [selectItems, setSelctItems] = useState([]);
  const [sort, setSort] = useState({
    sortBy: null,
    sortOrder: null,
    accessor: null,
  });

  const debouncedSearch = useDebounce(search);
  const previousSearch = usePrevious(debouncedSearch);

  const { data, status, isFetching, refetch } = useQuery({
    queryKey: [
      "processings-stats",
      {
        page,
        pageSize,
        search: debouncedSearch,
        dateFrom: dateRange[0],
        dateTo: dateRange[1],
      },
    ],
    queryFn: () => {
      const options = {
        offset: (page - 1) * pageSize,
        limit: pageSize,
        terms: debouncedSearch,
      };

      if (dateRange[0]) {
        Object.assign(options, { dateFrom: dateRange[0] });
      }

      if (dateRange[1]) {
        Object.assign(options, { dateTo: dateRange[1] });
      }

      if (debouncedSearch !== previousSearch) {
        setPage(1);
      }

      if (sort.sortBy && sort.sortOrder && sort.sortBy !== "none") {
        Object.assign(options, {
          sortBy: sort.sortBy,
          sortOrder: sort.sortOrder,
        });
      }

      if (selectItems.length > 0) {
        Object.assign(options, {
          statuses: selectItems,
        });
      }

      return getProcessingsPaged(options);
    },
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
    keepPreviousData: true,
  });

  const handleChangeDateRange = (range) => {
    setDateRange(range);
    setPage(1);
  };

  const handleClearFilter = () => {
    setPage(1);
    setSearch("");
    setDateRange([null, null]);
  };

  const handleSelectItems = (items) => {
    setPage(1);
    setSelctItems(items);
  };

  const handleHeaderClick = async (column) => {
    switch (column.sortDirection) {
      case "none":
        setSort({
          sortBy: "ASC",
          sortOrder: column.order,
          accessor: column.id,
        });
        break;
      case "ASC":
        setSort({
          sortBy: "DESC",
          sortOrder: column.order,
          accessor: column.id,
        });
        break;
      case "DESC":
        setSort({
          sortBy: "none",
          sortOrder: null,
          accessor: column.id,
        });
        break;
    }
    setPage(1);
  };

  useEffect(() => {
    refetch();
  }, [refetch, sort, selectItems]);

  const items = data?.results || [];

  return (
    <div className="processings-stats">
      <ProcessingsHeader search={search} setSearch={setSearch} />
      <div className="processings-stats__controls">
        <Pagination
          onChangePage={setPage}
          onChangePageSize={setPageSize}
          isLoading={isFetching}
          limit={pageSize}
          offset={(page - 1) * pageSize}
          total={data?.total || 0}
          showSinglePage
        />
        <Divider />
        <MultipleSelect items={selectItems} setItems={handleSelectItems} />
        <DateRangeInput2
          className="processings-stats__controls__date-range"
          {...DATE_FNS_FORMATS.ddMMyyyy}
          value={dateRange}
          dayPickerProps={{
            months: MONTHS,
            weekdaysShort: WEEKDAYS,
          }}
          onChange={handleChangeDateRange}
          shortcuts={DEFAULT_SHORTCUTS}
          closeOnSelection={false}
          highlightCurrentDay
          allowSingleDayRange
          // If enabled, it cause bug with shortcuts, changes dont correct update range in month view
          singleMonthOnly={false}
          startInputProps={{ placeholder: t`Start date` }}
          endInputProps={{ placeholder: t`End date` }}
        />
        <Button
          outlined
          disabled={dateRange.every((v) => v === null)}
          icon={IconNames.CROSS}
          onClick={() => handleChangeDateRange([null, null])}
        />
      </div>
      <ProcessingsContent
        hasFilters={search || page > 1 || dateRange.some(Boolean)}
        onClear={handleClearFilter}
        status={status}
        data={items}
        sort={sort}
        handleHeaderClick={handleHeaderClick}
      />
    </div>
  );
}

export default ProcessingsStats;
