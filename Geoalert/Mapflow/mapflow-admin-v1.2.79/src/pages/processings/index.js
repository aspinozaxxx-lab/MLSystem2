import React, { useState } from "react";
import { Trans, t } from "@lingui/macro";
import { H2, Callout, Intent, H3 } from "@blueprintjs/core";

import { Breadcrumbs } from "containers";
import { ProcessingList } from "containers";
import { useParams } from "react-router-dom";
import useProjectQuery from "components/project-workflow-list/queries";
import { EditableNameInput } from "components";

import { useProjectProgress } from "hooks/use-project-progress";
import ProjectProgress from "components/project-progress";
import ProjectDetailsDialog from "components/project-card/project-details-dialog";
import { UPDATE_PROJECT_NAME } from "shared/api/project";

function Processings() {
  const { projectId } = useParams();

  const [isOpenDetailsProject, setIsOpenDetailsProject] = useState(false);
  const handleClick = () => setIsOpenDetailsProject(!isOpenDetailsProject);

  const { projectProgressResult } = useProjectProgress(projectId);
  const { data } = useProjectQuery(projectId);

  return (
    <div className="processings">
      <Breadcrumbs />
      <div className="processings__container">
        <H2 className="workflows__name">
          <EditableNameInput
            value={data?.name}
            projectId={projectId}
            mutKey={["updateProjectName", projectId]}
            mutationVariables={{ projectId }}
            successMessage={t`Project name updated`}
            errorMessage={t`Error update project name`}
            field={"name"}
            mutRequest={UPDATE_PROJECT_NAME}
            refetchQueryKey={["project", "name", projectId]}
          />
        </H2>

        {data?.description && (
          <H3 className="workflows__name">
            <EditableNameInput
              value={data?.description}
              projectId={projectId}
              mutKey={["updateProjectDescription", projectId]}
              mutRequest={UPDATE_PROJECT_NAME}
              mutationVariables={{ projectId }}
              successMessage={t`Project description updated`}
              errorMessage={t`Error update project description`}
              field={"description"}
            />
          </H3>
        )}
        <H2>
          <Trans>Processings</Trans>
        </H2>
        <div className="callout">
          <Callout
            className="create-processing-tip"
            intent={Intent.PRIMARY}
            title={<Trans id="create-processing-tip">Tip</Trans>}
          >
            <Trans>
              To to create a processing please go to workflows and select one
            </Trans>
          </Callout>

          {projectProgressResult && (
            <Callout
              className="create-processing-progress"
              intent={Intent.PRIMARY}
              title={<Trans>Status</Trans>}
              icon={null}
            >
              <ProjectProgress
                id={projectId}
                progress={projectProgressResult}
                isProcessings
                handleOpen={handleClick}
              />
            </Callout>
          )}
        </div>
        <ProcessingList />
      </div>

      {isOpenDetailsProject && (
        <ProjectDetailsDialog projectId={projectId} handleClose={handleClick} />
      )}
    </div>
  );
}

export default React.memo(Processings);
