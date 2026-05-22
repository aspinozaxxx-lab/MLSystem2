import React from "react";

import { Trans } from "@lingui/macro";

import { EmptyMessage, ErrorMessage, StateLoading } from "components";

import Table from "components/table";

function WorkflowListContent({ status, workflowDefs, columns }) {
  if (status === "loading") {
    return (
      <StateLoading
        style={{ flex: 1 }}
        title={<Trans id="Fetching workflows" />}
      />
    );
  }

  if (status === "error") {
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Could not fetch workflows" />}
      />
    );
  }

  if (status === "success" && workflowDefs.length === 0) {
    return <EmptyMessage title={<Trans id="No workflow created yet" />} />;
  }

  if (status === "success" && workflowDefs.length > 0) {
    return <Table striped data={workflowDefs} columns={columns} showIndex />;
  }
}

export default React.memo(WorkflowListContent);
