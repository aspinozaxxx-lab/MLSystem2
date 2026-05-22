export default [
  {
    id: "c1526a8c-b9fd-4fc0-a5f2-cb5ca943c721",
    name: "Processing",
    created: "2020-12-22T11:37:15.765376Z[UTC]",
    updated: "2020-12-22T12:37:15.765386Z[UTC]",
    description: "Huge processing",
    progress: { status: "OK", percentCompleted: 100, __typename: "Progress" },
    workflowDef: {
      name: "wd-1",
      __typename: "WorkflowDef",
    },
    __typename: "Processing",
  },
  {
    id: "58eac08e-f07f-41ba-9885-e1b1483e1855",
    name: "Processing A",
    created: "2019-12-16T11:10:29.635932Z[UTC]",
    updated: "2019-12-16T12:10:29.635936Z[UTC]",
    description: "The processing",
    progress: {
      status: "UNPROCESSED",
      percentCompleted: 0,
      __typename: "Progress",
    },
    workflowDef: {
      name: "wd-2",
      __typename: "WorkflowDef",
    },
    __typename: "Processing",
  },
];
