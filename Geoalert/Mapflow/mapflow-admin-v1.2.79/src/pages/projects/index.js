import React, { useRef, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trans } from "@lingui/macro";
import { IconNames } from "@blueprintjs/icons";
import { Button, Intent } from "@blueprintjs/core";

import { useProjectSearch } from "app";

import StateLoading from "components/state-loading";
import ProjectCard from "components/project-card";
import EmptyMessage from "components/empty-message";
import ErrorMessage from "components/error-message";
import Pagination from "components/pagination";

import { POLL_INTERVAL } from "constants/envs";

import useDebounce from "hooks/use-debounce";
import usePrevious from "hooks/use-previous";

import { setTestId } from "test-utils/set-testid";

import ProjectsHeader from "./projects-header";
import { getProjects } from "./queries";
import { withCreateDialog } from "./with-create-dialog";
import ProjectDetailsDialog from "components/project-card/project-details-dialog";

const DEFAULT_PAGE_SIZE = 20;

function Projects({ onCreateProject }) {
  const totalRef = useRef(null);
  const previousTotal = totalRef.current;

  const { search } = useProjectSearch();
  const [pageSize, handleChangePageSize] = useState(DEFAULT_PAGE_SIZE);
  const [page, handleChangePage] = useState(1);

  const [projectId, setProjectId] = useState(null);

  const handleOpen = (id) => setProjectId(id);

  const handleClose = () => setProjectId(null);
  const debouncedSearch = useDebounce(search);
  const previousSearch = usePrevious(debouncedSearch);

  const shouldStopQuery = useMemo(() => {
    const currentOffset = (page - 1) * pageSize;
    return previousTotal && currentOffset > previousTotal;
  }, [page, pageSize, previousTotal]);

  useEffect(() => {
    if (shouldStopQuery) {
      const lastPageAtMoment = Math.ceil(previousTotal / pageSize);
      handleChangePage(lastPageAtMoment);
    }
  }, [handleChangePage, pageSize, previousTotal, shouldStopQuery]);

  const { data, status, isFetching, isInitialData } = useQuery({
    queryKey: ["projects", page, pageSize, debouncedSearch],
    queryFn: () => {
      const options = {
        offset: (page - 1) * pageSize,
        limit: pageSize,
        filter: debouncedSearch.trim(),
      };

      if (debouncedSearch !== previousSearch) {
        handleChangePage(1);
      }

      return getProjects(options);
    },
    refetchInterval: POLL_INTERVAL,
    refetchOnWindowFocus: false,
    keepPreviousData: true,
    enabled: !shouldStopQuery, // for prevent request with no actual page offset
  });

  if (status === "error")
    return (
      <ErrorMessage
        title={<Trans id="Error" />}
        description={<Trans id="Error fetch projects" />}
      />
    );

  if (status === "loading")
    return (
      <StateLoading
        className="projects-loader"
        title={<Trans id="Fetching Projects" />}
      />
    );

  const projects = data?.results || [];

  totalRef.current = data.total;

  const pagination = (
    <Pagination
      onChangePage={handleChangePage}
      onChangePageSize={handleChangePageSize}
      isLoading={isFetching}
      limit={pageSize}
      offset={(page - 1) * pageSize}
      total={data.total}
    />
  );

  if (projects.length === 0) {
    if (page === 1 && !isInitialData && debouncedSearch === "") {
      return (
        <EmptyMessage
          title={<Trans id="You haven’t created any project yet" />}
          description={
            <Trans id="To create a new project, click the button bellow." />
          }
          action={
            <Button
              large
              elementRef={setTestId`create-new-project`}
              icon={IconNames.ADD}
              intent={Intent.SUCCESS}
              text={<Trans id="Create project" />}
              onClick={onCreateProject}
            />
          }
        />
      );
    }

    return (
      <>
        <div className="projects">
          <ProjectsHeader onCreate={onCreateProject} />
          {pagination}
          <EmptyMessage title={<Trans id="Projects not found" />} />
        </div>
      </>
    );
  }

  return (
    <>
      <div className="projects">
        <ProjectsHeader onCreate={onCreateProject} />
        {pagination}
        <div className="project-list">
          {projects.map((project, index) => (
            <ProjectCard key={index} data={project} handleOpen={handleOpen} />
          ))}
        </div>

        {projectId && (
          <ProjectDetailsDialog
            projectId={projectId}
            handleClose={handleClose}
            handleOpen={handleOpen}
          />
        )}
      </div>
    </>
  );
}

export default withCreateDialog(Projects);
