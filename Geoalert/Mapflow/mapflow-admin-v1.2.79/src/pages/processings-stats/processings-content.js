import { Trans } from "@lingui/macro";
import ProcessingsTable from "./table";

import { StateLoading, ErrorMessage, EmptyMessage } from "components";
import { Button, Intent } from "@blueprintjs/core";

export const ProcessingsContent = ({
  hasFilters,
  onClear,
  status,
  data,
  sort,
  handleHeaderClick,
}) => {
  if (status === "error") {
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Error fetch processings" />}
      />
    );
  }

  if (status === "loading") {
    return (
      <StateLoading
        className="processing-loader"
        title={<Trans id="Fetching Processings" />}
      />
    );
  }

  if (status === "success" && data.length === 0 && hasFilters) {
    return (
      <EmptyMessage
        title={<Trans>No search results</Trans>}
        description={
          <Trans>
            Your search didn't match any processings. Try searching for
            something else or clear selected filters
          </Trans>
        }
        action={
          <Button onClick={onClear} intent={Intent.NONE}>
            <Trans>Clear filters</Trans>
          </Button>
        }
      />
    );
  }

  if (status === "success" && data.length === 0) {
    return <EmptyMessage title={<Trans id="No processings started yet" />} />;
  }

  return (
    <ProcessingsTable
      data={data}
      sort={sort}
      handleHeaderClick={handleHeaderClick}
    />
  );
};
