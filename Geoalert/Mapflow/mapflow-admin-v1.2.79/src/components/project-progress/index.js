import { Intent, Position, Tag, Tooltip } from "@blueprintjs/core";
import { Trans } from "@lingui/macro";
import { STATUS_PENDING, STATUS_SUCCESS } from "constants/common";
import { getProcessETAWithDate } from "utils/getProcessETA";

const ProjectProgress = ({
  progress,
  id,
  handleOpen = () => {},
  isProcessings,
}) => {
  const successProcessingsCount = progress?.details?.find(
    (detail) => detail.status === STATUS_SUCCESS,
  )?.count;

  return (
    <Tooltip
      position={Position.BOTTOM}
      content={<Trans>Click to open project processings details</Trans>}
    >
      <div
        onClick={(e) => {
          e.stopPropagation();
          handleOpen(id);
        }}
        style={{ width: "fit-content", cursor: "pointer" }}
      >
        {progress.status !== STATUS_PENDING && successProcessingsCount >= 0 && (
          <Tag intent={Intent.SUCCESS} minimal round>
            <span>
              <Trans>Successful processings: </Trans> {successProcessingsCount}
            </span>
          </Tag>
        )}

        {progress.status === STATUS_PENDING && (
          <Tag intent={Intent.WARNING} minimal round>
            <span>
              <Trans>Completion estimated time:</Trans>{" "}
              {isProcessings && <br />}
              {progress.estimate > 0 && progress.estimate ? (
                getProcessETAWithDate(progress.estimate)
              ) : (
                <Trans id="No data" />
              )}
            </span>
          </Tag>
        )}
      </div>
    </Tooltip>
  );
};

export default ProjectProgress;
