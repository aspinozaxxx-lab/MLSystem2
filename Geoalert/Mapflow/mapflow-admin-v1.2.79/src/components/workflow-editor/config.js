import * as monaco from "monaco-editor";

window.MonacoEnvironment = {
  getWorker(moduleId, label) {
    switch (label) {
      case "editorWorkerService":
        return new Worker(
          new URL("monaco-editor/esm/vs/editor/editor.worker", import.meta.url),
        );
      case "yaml":
        return new Worker(new URL("monaco-yaml/yaml.worker", import.meta.url));
      default:
        throw new Error(`Unknown label ${label}`);
    }
  },
};

monaco.languages.registerHoverProvider("yaml", {
  provideHover: (model, positon, token) => {
    const word = model.getWordAtPosition(positon);
    if (word && word.word.startsWith("FILE_")) {
      return {
        range: new monaco.Range(
          word.startColumn,
          positon.lineNumber,
          word.startColumn,
          positon.lineNumber,
        ),
        contents: [{ value: "Alt + click for navigate to file, if tab exist" }],
      };
    }
  },
});
