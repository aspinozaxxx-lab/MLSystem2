import React from "react";
import classnames from "classnames";
import { useLingui } from "@lingui/react";
import { t, Trans } from "@lingui/macro";
import { Classes, Button, Intent, H5, Tag } from "@blueprintjs/core";
import { Text, Card, H4, Elevation } from "@blueprintjs/core";
import { Tooltip, Position } from "@blueprintjs/core";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import useClipboard from "react-use-clipboard";

import { useGoTo } from "hooks/use-go-to";
import * as routes from "constants/routes";
import { IconNames } from "@blueprintjs/icons";
import ConfirmDialog from "components/confirm-dialog";
import { deleteProject } from "./queries";
import { showToast, getSuccessToast, getErrorToast } from "toaster";
import { setTestId } from "test-utils/set-testid";
import { GET_PROJECTS } from "pages/projects/queries";
import { generatePath } from "react-router-dom";
import ProjectProgress from "components/project-progress";
import ToLocaleTime from "components/toLocalTime";

function ProjectCard({ data, handleOpen }) {
  const { id, name, description, created, updated, user, progress } = data;
  const queryClient = useQueryClient();

  const goToProject = useGoTo(routes.PROJECT_WORKFLOWS, {
    projectId: id,
  });

  const mutation = useMutation(deleteProject, {
    onSuccess: () => {
      queryClient.invalidateQueries(GET_PROJECTS, { force: true });
      showToast(
        getSuccessToast(t`Project "${name}" deleted`, {
          icon: IconNames.TRASH,
        }),
      );
    },
    onError: () => showToast(getErrorToast(t`Error delete project`)),
  });

  const [isCopied, setCopied] = useClipboard(user.email, {
    successDuration: 1700,
  });

  const handleNewTabClick = (e) => {
    e.stopPropagation();
    window.open(
      generatePath(routes.PROJECT_WORKFLOWS, { projectId: id }),
      "_blank",
    );
  };

  return (
    <Card
      interactive={false}
      className="project-card"
      elevation={Elevation.ONE}
      onClick={goToProject}
    >
      <ConfirmDialog
        className="project-card__remove-btn"
        intent={Intent.DANGER}
        icon={IconNames.TRASH}
        confirmButtonText={<Trans id="Delete" />}
        cancelButtonText={<Trans id="Cancel" />}
        text={
          <H5>
            <Trans id="Confirm delete project" />
          </H5>
        }
        onConfirm={(close) => {
          mutation.mutate(id);
          close();
        }}
      >
        {({ showDialog }) => (
          <Button
            minimal
            elementRef={setTestId`delete-project`}
            intent={Intent.DANGER}
            icon={IconNames.TRASH}
            loading={mutation.isLoading}
            onClick={showDialog}
          />
        )}
      </ConfirmDialog>
      <div className="project-card__open-new-tab">
        <Button
          icon={IconNames.SHARE}
          intent={Intent.PRIMARY}
          minimal
          onClick={handleNewTabClick}
        />
      </div>
      <div>
        <H4 className="project-card__name">
          <Text className="project-card__name" ellipsize>
            {name}
          </Text>
        </H4>
        <Text
          className={classnames(
            "project-card__description",
            Classes.TEXT_MUTED,
          )}
        >
          {description}
        </Text>
      </div>

      <div>
        <Tooltip
          position={Position.BOTTOM}
          content={<div>{isCopied ? t`Copied!` : t`Click to copy`}</div>}
        >
          <div
            className="project-card__email"
            onClick={(e) => {
              e.stopPropagation();
              setCopied();
            }}
          >
            {user?.name || user?.preferredUsername || user?.email}
          </div>
        </Tooltip>
      </div>
      <div className={classnames("project-card__date", Classes.TEXT_MUTED)}>
        <Tooltip
          position={Position.BOTTOM}
          content={
            <span className="date-time">
              <Trans>Created</Trans> {<ToLocaleTime time={created} />}
            </span>
          }
        >
          <span className="date-time">
            <Trans>Updated</Trans> {<ToLocaleTime time={updated} />}
          </span>
        </Tooltip>
      </div>
      <ProjectProgress id={id} handleOpen={handleOpen} progress={progress} />
    </Card>
  );
}

export default React.memo(ProjectCard);
