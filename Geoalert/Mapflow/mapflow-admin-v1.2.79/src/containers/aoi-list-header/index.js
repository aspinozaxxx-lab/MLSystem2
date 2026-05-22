import React, {useState} from "react";
import { Trans } from "@lingui/macro";
import { useQueryClient } from "@tanstack/react-query";
import classnames from "classnames";

import { SortIndicator } from "components";

const findField = (_) => ({ field }) => field === _;
const excludeField = (_) => ({ field }) => field !== _;

const getSortState = (sort) => (id) => {
  const state = sort.find(findField(id));
  const isSorted = Boolean(state);
  if (!isSorted) return { isSorted, sortDirection: null };
  return { isSorted, sortDirection: state.desc ? "DESC" : "ASC" };
};

const updateSortState = (field, event) => (sort) => {
  const sortItem = sort.find(findField(field));
  const desc = sortItem ? !sortItem.desc : false;
  if (event.ctrlKey || event.metaKey) {
    return sort.filter(excludeField(field));
  } else if (!event.shiftKey) {
    return [].concat({ field, desc });
  } else {
    return sort.filter(excludeField(field)).concat({ field, desc });
  }
};

function AoiListHeaderColl({ id, className, label, onClick, getState }) {
  const { isSorted, sortDirection } = getState(id);
  return (
    <div className={className} onClick={(e) => onClick(id, e)}>
      <span className={classnames("aoi-list-header-coll")}>
        <span className="aoi-list-header-coll__title">{label}</span>
        <SortIndicator
          isSorted={isSorted}
          sortDirection={sortDirection}
          className="aoi-list-header-coll__sort-indicator"
        />
      </span>
    </div>
  );
}

function AoiListHeader() {
  const client = useQueryClient()

  const rememberSortState = (sort) => {
    client.setQueryData(["aoiListSort"], sort);
    return sort;
  }

  const [sort, setSort] = useState([])

  const handleSort = (id, e) => setSort(rememberSortState(updateSortState(id, e)(sort)));

  return (
    <div className="aoi-list__header">
      <div className="aoi-list-item">
        <AoiListHeaderColl
          id="area"
          className="aoi-item-coll area"
          label={<Trans>Area</Trans>}
          onClick={handleSort}
          getState={getSortState(sort)}
        />
        <AoiListHeaderColl
          id="percentCompleted"
          className="aoi-item-coll percent-completed"
          label={<Trans>Progress</Trans>}
          onClick={handleSort}
          getState={getSortState(sort)}
        />
        <AoiListHeaderColl
          id="status"
          className="aoi-item-coll status"
          label={<Trans>Status</Trans>}
          onClick={handleSort}
          getState={getSortState(sort)}
        />
      </div>
    </div>
  );
}

export default React.memo(AoiListHeader);