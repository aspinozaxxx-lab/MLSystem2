import React from "react";
import { Trans, t } from "@lingui/macro";
import { H2, H3 } from "@blueprintjs/core";

import LinkWorkflowButton from "components/link-workflow-button";
import { Breadcrumbs } from "containers";
import {
  ProjectWorkflowList,
  StateLoading,
  ErrorMessage,
  EditableNameInput,
} from "components";
import { useParams } from "react-router-dom";
import CreateAndLinkWorkflow from "./create-and-link-workflow";
import useProjectQuery from "components/project-workflow-list/queries";
import { AdminRender } from "providers/auth/use-user";
import { UPDATE_PROJECT_NAME } from "shared/api/project";

function ProjectWorkflows() {
  const { projectId } = useParams();

  const { data, isLoading, isSuccess, isError } = useProjectQuery(projectId);

  return (
    <div className="workflows">
      <Breadcrumbs />
      <div className="workflows__container">
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

        <div className="workflows__header">
          <H3 className="workflows__header-title">
            <Trans>Project Workflows</Trans>
          </H3>

          <AdminRender>
            <div className="workflows__header-action">
              {isLoading && <StateLoading />}
              {isSuccess && !data?.isDefault && (
                <>
                  <LinkWorkflowButton projectId={projectId} />
                  <CreateAndLinkWorkflow />
                </>
              )}
            </div>
          </AdminRender>
        </div>

        {isError && (
          <ErrorMessage
            title={<Trans id="Error" />}
            description={<Trans id="Error fetch projects workflows" />}
          />
        )}

        {isSuccess && <ProjectWorkflowList />}
      </div>
    </div>
  );
}

export default React.memo(ProjectWorkflows);
