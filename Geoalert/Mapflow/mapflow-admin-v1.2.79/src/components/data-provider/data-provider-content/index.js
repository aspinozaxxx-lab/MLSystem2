import React from "react";
import { Trans } from "@lingui/macro";
import StateLoading from "components/state-loading";
import ErrorMessage from "components/error-message";
import EmptyMessage from "components/empty-message";
import Table from "components/table";

function DataProviderContent({ status, data, columns }) {

  if (status === "loading") {
    return (
      <StateLoading
        style={{ flex: 1 }}
        title={<Trans id="Fetching data providers" />}
      />
    );
  }

  if (status === "error") {
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Could not fetch data providers" />}
      />
    );
  }

  if (status === "success" && data.length === 0) {
    return <EmptyMessage title={<Trans id="No data provider created yet" />} />;
  }

  if (status === "success" && data.length > 0) {
    return <Table striped data={data} columns={columns} showIndex />;
  }
}

export default React.memo(DataProviderContent);
