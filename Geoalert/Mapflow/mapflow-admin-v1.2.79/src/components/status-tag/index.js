import React from "react";
import { Tag } from "@blueprintjs/core";
import { t } from "@lingui/macro";

import { ProgressStatuses, STATUS_PENDING } from "constants/common";
import { getProcessETA } from "utils/getProcessETA";
const { T, I } = ProgressStatuses;

function StatusTag({ statusCode, percent, estimate, withPercent = null }) {
  return (
    <Tag minimal round intent={I[statusCode]}>
      {t({ id: T[statusCode] })}
      {statusCode === STATUS_PENDING && withPercent && <> / {percent}% </>}
      {statusCode === STATUS_PENDING && estimate > 0 && (
        <> ETA:{getProcessETA(estimate)}</>
      )}
    </Tag>
  );
}

export default React.memo(StatusTag);
