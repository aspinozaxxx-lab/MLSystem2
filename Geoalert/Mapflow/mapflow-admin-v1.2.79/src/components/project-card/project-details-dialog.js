import React, { memo } from "react";
import { Trans } from "@lingui/macro";
import { Dialog, DialogBody } from "@blueprintjs/core";

import FormatArea from "components/format-area";
import StatusTag from "components/status-tag";
import { useProjectProgress } from "hooks/use-project-progress";
import Table from "components/table";
import ToLocaleTime from "components/toLocalTime";

function ProjectDetailsDialog({ projectId, handleClose }) {
  const { projectProgressResult } = useProjectProgress(projectId);

  const details = projectProgressResult ? projectProgressResult.details : null;

  const columns = [
    {
      Header: <Trans id="Status" />,
      accessor: "status",
      id: "status",
      Cell: ({ cell }) => {
        return <StatusTag statusCode={cell.value} />;
      },
    },
    {
      Header: <Trans id="Count" />,
      id: "count",
      accessor: "count",
      Cell: ({ cell }) => console.log(cell) || <span>{cell.value}</span>,
    },
    {
      Header: <Trans id="Area" />,
      id: "area",
      accessor: "area",
      Cell: ({ cell }) => <FormatArea cutZeros area={cell.value} />,
    },
    {
      Header: <Trans id="Updated" />,
      id: "statusUpdateDate",
      accessor: "statusUpdateDate",
      Cell: ({ cell }) => {
        if (!cell.value) {
          return "-";
        }
        return <ToLocaleTime time={cell.value} />;
      },
    },
  ];

  if (!details || !details.length) {
    return null;
  }

  return (
    <div
      className="aoi-messages-container"
      onClick={(e) => e.stopPropagation()}
    >
      <Dialog
        className="details-dialog"
        icon="info-sign"
        isOpen={true}
        onClose={(event) => {
          event.stopPropagation();
          handleClose(null);
        }}
        title={<Trans>Project Details</Trans>}
        autoFocus={true}
        canEscapeKeyClose={true}
        canOutsideClickClose={true}
        enforceFocus={true}
        usePortal={true}
      >
        <DialogBody className="dialog-body">
          <Table striped interactive data={details} columns={columns} />
        </DialogBody>
      </Dialog>
    </div>
  );
}

export default memo(ProjectDetailsDialog);
