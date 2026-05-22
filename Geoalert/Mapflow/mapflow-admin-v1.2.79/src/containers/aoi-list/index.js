import React from "react";
import { Trans } from "@lingui/macro";

import { StateLoading, EmptyMessage, ErrorMessage } from "components";

import { AoiListItem } from "./aoi-list-item";
import { AoiListHeader } from "containers";
import { Button, Intent } from "@blueprintjs/core";

function AoiList({ onUpload, isGeotiffDataProvider, aoiResult }) {
  const { data, isLoading, isError, error } = aoiResult;

  const isEmpty = !(data && data.length > 0);

  const loadingMessage = (
    <StateLoading
      className="aoi-list-loader"
      title={<Trans>Loading</Trans>}
      description={<Trans>Fetching AOI's</Trans>}
    />
  );

  const errorMessage = (
    <ErrorMessage
      title={<Trans>Unable to load AOIs</Trans>}
      description={error}
    />
  );

  const emptyMessage = (
    <EmptyMessage
      title={<Trans>You haven’t created any AOI yet</Trans>}
      action={
        <Button
          outlined
          intent={Intent.SUCCESS}
          text={<Trans>Upload Areas of interest</Trans>}
          onClick={onUpload}
        />
      }
    />
  );

  return (
    <div className="aoi-list">
      <AoiListHeader />
      {isLoading ? (
        loadingMessage
      ) : isError ? (
        errorMessage
      ) : isEmpty ? (
        emptyMessage
      ) : (
        <>
          {data.map((aoi) => (
            <AoiListItem
              isGeotiffDataProvider={isGeotiffDataProvider}
              id={aoi.id}
              key={aoi.id}
            />
          ))}
        </>
      )}
    </div>
  );
}

export default React.memo(AoiList);
