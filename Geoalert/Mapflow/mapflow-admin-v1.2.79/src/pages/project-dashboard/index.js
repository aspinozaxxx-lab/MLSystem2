import React, { useCallback, useMemo } from "react";

import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { gql, useApolloClient } from "@apollo/client";

import { H2, Intent, Position, Tooltip } from "@blueprintjs/core";

import { Breadcrumbs } from "containers";
import { POLL_INTERVAL } from "constants/envs";
import StateLoading from "components/state-loading";
import { UTCRemove } from "utils/date";
import { Trans, useLingui } from "@lingui/react";
import { FormatArea } from "components";
import StatsItem from "components/stats-item";

export const GET_PROJECT_STATS = gql`
  query getProjectStats($projectId: ID!) {
    project(id: $projectId) {
      name
      description
      created
      updated
      aoiCount
      area
      progress {
        status
        percentCompleted
        details {
          status
          count
        }
      }
    }
  }
`;

const getProjectStats = (client) => async (...args) => {
  const [, projectId] = args;
  const result = await client.query({
    query: GET_PROJECT_STATS,
    fetchPolicy: "no-cache",
    variables: { projectId },
  });

  return result?.data?.project;
};

const groupDetailsProjectCountByStatus = (data) => {
  return data?.progress?.details.reduce(
    (acc, { count, status }) => ({
      ...acc,
      [status]: count,
    }),
    {},
  );
};

let dateFormatYMD = {
  year: "numeric",
  month: "numeric",
  day: "numeric",
};

let dateFormatHM = {
  hour: "2-digit",
  minute: "2-digit",
};

function ProjectDashboard() {
  const { projectId } = useParams();

  const { i18n } = useLingui();

  const client = useApolloClient();
  const { data, status } = useQuery({
    queryKey: ["project-stats", projectId],
    queryFn: () => getProjectStats(client),
    refetchInterval: POLL_INTERVAL,
  });

  const progressByStatus = useMemo(
    () => groupDetailsProjectCountByStatus(data),
    [data],
  );

  if (status === "loading") {
    return (
      <StateLoading
        className="project_dashboard__loader"
        title={<Trans id="Dashboard loading" />}
      />
    );
  }

  return (
    <>
      <Breadcrumbs />
      <div className="project_dashboard">
        <H2>{data.name}</H2>
        {data.description && (
          <p className="project_dashboard__description">{data.description}</p>
        )}
        <ProgressBar {...progressByStatus} />
        <div style={{ marginTop: 16 }} className="project_dashboard__tags">
          <StatsItem
            title={<Trans id="Updated" />}
            topLeft={i18n.date(UTCRemove(data.updated), dateFormatYMD)}
            topRight={i18n.date(UTCRemove(data.updated), dateFormatHM)}
          />

          <StatsItem
            title={<Trans id="Created" />}
            topLeft={i18n.date(UTCRemove(data.created), dateFormatYMD)}
            topRight={i18n.date(UTCRemove(data.created), dateFormatHM)}
          />

          <StatsItem title={<Trans id="Aoi count" />}>
            {data.aoiCount}
          </StatsItem>
          <StatsItem title={<Trans id="Aoi area" />}>
            <FormatArea area={data.area} />
          </StatsItem>

          <StatsItem title={<Trans id="Completed" />}>
            {data?.progress?.percentCompleted}%
          </StatsItem>
        </div>
      </div>
    </>
  );
}

const calcPart = (total) => (value) => (value / total) * 100 + "%";

const ProgressBar = ({
  OK = 0,
  UNPROCESSED = 0,
  IN_PROGRESS = 0,
  FAILED = 0,
}) => {
  const total = OK + UNPROCESSED + IN_PROGRESS + FAILED;

  const calcPercent = useCallback((value) => calcPart(total)(value), [total]);

  return (
    <div className="project_dashboard__progressbar">
      <ProgressBarItem
        width={calcPercent(OK)}
        tooltipContent={<Trans id="OK" />}
        status="OK"
      />

      <ProgressBarItem
        width={calcPercent(UNPROCESSED)}
        tooltipContent={<Trans id="UNPROCESSED" />}
        status="UNPROCESSED"
      />

      <ProgressBarItem
        width={calcPercent(IN_PROGRESS)}
        tooltipContent={<Trans id="IN_PROGRESS" />}
        status="IN_PROGRESS"
      />

      <ProgressBarItem
        width={calcPercent(FAILED)}
        tooltipContent={<Trans id="FAILED" />}
        status="FAILED"
      />
    </div>
  );
};

const ProgressBarItem = ({ width, tooltipContent, status }) => {
  return (
    <div
      className={`project_dashboard__progressbar-${status}`}
      style={{ width }}
    >
      <Tooltip
        className={`project_dashboard__progressbar__tooltip`}
        intent={Intent.PRIMARY}
        position={Position.BOTTOM}
        content={tooltipContent}
        hoverOpenDelay={20}
      >
        <div />
      </Tooltip>
    </div>
  );
};

export default React.memo(ProjectDashboard);
