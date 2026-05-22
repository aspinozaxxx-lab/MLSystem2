export const MAIN = "/";

export const WORKFLOWS = "/workflows";
export const WORKFLOW_CREATE = `${WORKFLOWS}/create`;
export const WORKFLOW_EDIT = `${WORKFLOWS}/:workflowDefId/edit`;
export const WORKFLOW_USERS = `${WORKFLOWS}/:workflowDefId/users`;

export const PROJECTS = "/projects";
export const PROJECT = `${PROJECTS}/:projectId`;
export const PROJECT_WORKFLOWS = `${PROJECTS}/:projectId/workflows`;
export const PROJECT_WORKFLOW_LINK = `${PROJECTS}/:projectId/workflows/link`;
export const PROJECT_PROCESSINGS = `${PROJECTS}/:projectId/processings`;
export const PROJECT_SETTINGS = `${PROJECTS}/:projectId/settings`;
export const PROJECT_PROCESSING = `${PROJECTS}/:projectId/processings/:processingId`;

// Data Providers
export const DATA_PROVIDERS = "/data-providers";
export const DATA_PROVIDER_CREATE = `${DATA_PROVIDERS}/create`;
export const DATA_PROVIDER_EDIT = `${DATA_PROVIDERS}/:dataProviderId/edit-data-provider`;
export const DATA_PROVIDER_USERS = `${DATA_PROVIDERS}/:dataProviderId/users`;

export const PROCESSING_STATS = "/processings-stats";

export const ALL_ROUTES = [
  WORKFLOWS,
  WORKFLOW_CREATE,
  WORKFLOW_EDIT,
  WORKFLOW_USERS,

  PROJECTS,
  PROJECT,
  PROJECT_WORKFLOWS,
  PROJECT_WORKFLOW_LINK,
  PROJECT_PROCESSINGS,
  PROJECT_SETTINGS,
  PROJECT_PROCESSING,

  // Data Providers
  DATA_PROVIDERS,
  DATA_PROVIDER_CREATE,
  DATA_PROVIDER_EDIT,
  DATA_PROVIDER_USERS,

  PROCESSING_STATS,
];
