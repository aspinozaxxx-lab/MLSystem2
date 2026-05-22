import React, { useMemo } from "react";
import {
  matchPath,
  useLocation,
  useHistory,
  useParams,
} from "react-router-dom";
import { parse } from "path-to-regexp";
import { t, Trans } from "@lingui/macro";
import { Breadcrumbs as Bp3Breadcrumbs, Breadcrumb } from "@blueprintjs/core";
import { useQuery } from "@tanstack/react-query";
import pathOr from "ramda/src/pathOr";
import * as routes from "constants/routes";
import { useSkeleton } from "hooks/use-skeleton";
import { IconNames } from "@blueprintjs/icons";
import { gql, useApolloClient } from "@apollo/client";
import { GET_WORKFLOW } from "pages/create-edit-workflow";
import { GET_DATA_PROVIDER } from "components/data-provider/queries";

export const GET_PROJECT_NAME = gql`
  query getProjectName($id: ID!) {
    project(id: $id) {
      name
    }
  }
`;

export const GET_PROCESSING_NAME = gql`
  query getProcessingName($id: ID!) {
    processing(id: $id) {
      name
    }
  }
`;

function Breadcrumbs({ className = "" }) {
  const history = useHistory();
  const match = useLocationMatch();

  const { projectId, processingId, workflowDefId, dataProviderId } = useParams();

  const workflowName = useNameFromId(
    workflowDefId,
    ["workflowDef", "name"],
    GET_WORKFLOW,
  );

  const projectName = useNameFromId(
    projectId,
    ["project", "name"],
    GET_PROJECT_NAME,
  );
  const processingName = useNameFromId(
    processingId,
    ["processing", "name"],
    GET_PROCESSING_NAME,
  );


  const dataProviderName = useNameFromId(
    dataProviderId,
    ['dataProviders', 0, 'name'],
    GET_DATA_PROVIDER,
  );

  const names = useMemo(
    () => ({
      projectId: projectName,
      processingId: processingName,
      workflowDefId: workflowName,
      dataProviderId: dataProviderName
    }),
    [projectName, processingName, workflowName, dataProviderName],
  );

  // const loading = useMemo(() => {
  //   const mustLoads = [];
  //   for (const name in names)
  //     if (match.params[name]) mustLoads.push(names[name]);
  //   return !mustLoads.every(Boolean);
  // }, [names, match.params]);

  const { skeletoned } = useSkeleton(false);

  const breadcrumbs = buildBreadcrumbs(match, names, history);

  return (
    <div className={`breadcrumbs-container ${className}`}>
      <Bp3Breadcrumbs
        className={skeletoned`breadcrumbs`}
        currentBreadcrumbRenderer={renderCurrentBreadcrumb}
        items={breadcrumbs}
      />
    </div>
  );
}

export default React.memo(Breadcrumbs);

const ICON = IconNames.FOLDER_OPEN;

const PATH = {
  WORKFLOW_CREATE: "/workflows/create",
  WORKFLOW_EDIT: "/edit",
  PROJECTS: "/projects",
  WORKFLOWS: "/workflows",
  WORKFLOWS_USERS: "/users",
  PROJECT_PROCESSINGS: "/processings",
  PROJECT_WORKFLOW_LINK: "/workflows/link",
  DATA_PROVIDERS: "/data-providers",
  DATA_PROVIDER_CREATE: "/data-providers/create",
  DATA_PROVIDER_EDIT: "/edit-data-provider",
  DATA_PROVIDER_USERS: "/data-provider/users"
};

const BREADCRUMBS = {
  [PATH.WORKFLOW_CREATE]: [t`Workflows`, t`Create a Workflow`],
  [PATH.WORKFLOW_EDIT]: [t`Edit Workflow`],
  [PATH.PROJECTS]: [t`Projects`],
  [PATH.WORKFLOWS]: [t`Workflows`],
  [PATH.WORKFLOWS_USERS]: [t`Manage users`],
  [PATH.PROJECT_PROCESSINGS]: [t`Processings`],
  [PATH.PROJECT_WORKFLOW_LINK]: [t`Workflows`, t`Link workflows`],
  [PATH.DATA_PROVIDERS]: [t`Data Providers`],
  [PATH.DATA_PROVIDER_CREATE]: [t`Data Providers`, t`Create Data Provider`],
  [PATH.DATA_PROVIDER_EDIT]: [t`Edit Data Provider`],
  [PATH.DATA_PROVIDER_USERS]: [t`Manage users`],
};


const ROUTES = [
  routes.WORKFLOWS,
  routes.WORKFLOW_CREATE,
  routes.WORKFLOW_EDIT,
  routes.WORKFLOW_USERS,
  routes.PROJECT_WORKFLOWS,
  routes.PROJECT_PROCESSINGS,
  routes.PROJECT_PROCESSING,
  routes.PROJECT_SETTINGS,
  routes.PROJECT_WORKFLOW_LINK,
  routes.DATA_PROVIDERS,
  routes.DATA_PROVIDER_CREATE,
  routes.DATA_PROVIDER_USERS,
  routes.DATA_PROVIDER_EDIT
];

function buildBreadcrumbs(match, names, history) {
  if (!match) return [];
  const { path, url } = match;
  const parsed = parse(path);

  function processNestedRoutes(url) {
    const parts = url.split("/");
    if (parts[0] === "") return "/" + parts[1];
    return "/";
  }

  function addBreadcrumbs(breadcrumbs, key, index) {
    let toAdd;
    if (typeof key === "object") toAdd = [{ text: names[key.name] || "-" }];
    else if (typeof key === "string") {
      const transIds = BREADCRUMBS[key] || [
        <Trans id="breadcrumbs.fallback">-</Trans>,
      ];
      toAdd = transIds.map((id) => ({ text: <Trans id={id} /> }));
    }
    const link =
      index === 0
        ? processNestedRoutes(url)
        : url
            .split("/")
            .slice(0, index + 2)
            .join("/");
    const addProps = (b) => ({
      ...b,
      icon: ICON,
      onClick: () => {
        history.push(link);
      },
      className: "breadcrumb",
    });
    return breadcrumbs.concat(toAdd.map(addProps));
  }

  return parsed.reduce(addBreadcrumbs, []);
}

export function useNameFromId(id, path, query) {
  const client = useApolloClient();

  const getName = (query) => async (id) => {
    const result = await client.query({
      fetchPolicy: "no-cache",
      variables: { id },
      query,
    });

     return result?.data;
  };

  const getNameFromQuery = getName(query);
  const { data } = useQuery({
    queryKey: [...path, id],
    queryFn: () => getNameFromQuery(id),
    enabled: !!id,
  });


  const name = pathOr(null, path, data);
  return useMemo(() => name, [name]);
}

function useLocationMatch() {
  const { pathname } = useLocation();
  const matched = useMemo(() => {
    let match = null;
    for (let p of ROUTES) {
      match = matchPath(pathname, { path: p, exact: true });
      if (match) break;
    }
    return match;
  }, [pathname]);
  return matched;
}

const renderCurrentBreadcrumb = ({
  text,
  onClick: _,
  icon: __,
  ...restProps
}) => {
  // customize rendering of last breadcrumb
  return <Breadcrumb {...restProps}>{text}</Breadcrumb>;
};
