export default {
  id: "58eac08e-f07f-41ba-9885-e1b1483e1855",
  name: "Processing A",
  description: "Processing description",
  aoiCount: 0,
  bbox:
    "[37.545500524806094, 55.70438629847626, 37.672684002731735, 55.82583919374977]",
  vectorLayer: {
    id: "b33de484-af1f-4d52-ae46-efdad5cf3954",
    name: "Main vector layer",
    externalId: "21a1c544-d5f6-41c0-814b-d836c1f34ec9",
    tileJsonUrl: "https://tiles.com/123.json",
    __typename: "VectorLayer",
  },
  progress: {
    percentCompleted: 100,
    details: [{ status: "OK", area: 51985800, __typename: "ProgressDetail" }],
    __typename: "Progress",
  },
  workflowDef: {
    name: "Test",
    description: "Some experimental workflow",
    __typename: "WorkflowDef",
  },
  __typename: "Processing",
};
