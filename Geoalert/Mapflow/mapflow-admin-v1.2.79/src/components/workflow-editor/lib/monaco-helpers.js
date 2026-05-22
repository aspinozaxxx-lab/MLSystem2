import { editor, MarkerSeverity } from "monaco-editor";

const pickModelErrors = ({ severity }) => MarkerSeverity.Error === severity;

export const getMonacoErrors = () => {
  return editor.getModelMarkers({ owner: "yaml" }).filter(pickModelErrors);
};

export const hasModelErrorsByResourse = (uri) => {
  return editor.getModelMarkers({ resource: uri }).some(pickModelErrors);
};
