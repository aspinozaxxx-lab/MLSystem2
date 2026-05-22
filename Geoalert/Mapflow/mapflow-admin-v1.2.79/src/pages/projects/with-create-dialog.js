import React, { useState } from "react";
import { t } from "@lingui/macro";

import { showToast, getSuccessToast, getErrorToast } from "toaster";
import { useGoTo } from "hooks/use-go-to";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CreateProjectDialog } from "components/create-project-dialog";
import * as routes from "constants/routes";
import { postProject, GET_PROJECTS } from "./queries";

export const withCreateDialog = (Component) => (props) => {
  const queryClient = useQueryClient()
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const onCreateProject = () => setShowCreateDialog(true);
  const handleClose = () => setShowCreateDialog(false);

  const goToProject = useGoTo(routes.PROJECT_WORKFLOWS);
  const mutation = useMutation(postProject, {
    onSuccess: (data) => {
      setTimeout(() => {
        queryClient.invalidateQueries(GET_PROJECTS);
      }, 0);
      showToast(
        getSuccessToast(t`Project "${data.name}" successfully created`),
      );
      goToProject({ projectId: data.id });
    },
    onError: () => showToast(getErrorToast(t`Error creating project`)),
  });
  
  return (
    <>
      <CreateProjectDialog
        isOpen={showCreateDialog}
        handleClose={handleClose}
        createProject={mutation.mutate}
        isLoading={mutation.isLoading}
      />
      <Component {...props} onCreateProject={onCreateProject} />
    </>
  );
};
