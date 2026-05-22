export const cacheRedirects = {
  Query: {
    workflowInfo: (_, args, { getCacheKey }) =>
      getCacheKey({ __typename: "WorkflowDef", id: args.id }),
    project: (_, args, { getCacheKey }) =>
      getCacheKey({ __typename: "Project", id: args.id }),
    processing: (_, args, { getCacheKey }) =>
      getCacheKey({ __typename: "Processing", id: args.id }),
    aoiDetails: (_, args, { getCacheKey }) =>
      getCacheKey({ __typename: "Aoi", id: args.id }),
  },
};
