export const workflowInfoRedirectMock = jest.fn((_, args, { getCacheKey }) =>
  getCacheKey({ __typename: "WorkflowDef", id: args.id }),
);
export const projectRedirectMock = jest.fn((_, args, { getCacheKey }) =>
  getCacheKey({ __typename: "Project", id: args.id }),
);
export const processingRedirectMock = jest.fn((_, args, { getCacheKey }) =>
  getCacheKey({ __typename: "Processing", id: args.id }),
);

export const cacheRedirects = {
  Query: {
    workflowInfo: workflowInfoRedirectMock,
    project: projectRedirectMock,
    processing: processingRedirectMock,
  },
};
