import { useCallback, useEffect, useRef, useState } from "react";
import { Trans, t } from "@lingui/macro";

import { Uri } from "monaco-editor";
import * as monaco from "monaco-editor";

import { configureMonacoYaml } from "monaco-yaml";
import classNames from "classnames";
import { useTheme } from "hooks/use-theme";
import { ThemeNames } from "constants/common";
import { parseWorkflowYaml } from "./lib/parse-workflow-yaml";
import { getWordAt } from "./lib/get-word-at";
import { Button, Intent } from "@blueprintjs/core";
import { encode } from "base-64";

import "./config";
import { getErrorToast, showToast } from "toaster";
import { isEmptyYaml } from "./lib/yaml";
import {
  getMonacoErrors,
  hasModelErrorsByResourse,
} from "./lib/monaco-helpers";

// eslint-disable-next-line no-unused-vars
const monacoYaml = configureMonacoYaml(monaco, {
  enableSchemaRequest: true,
  schemas: [
    /** defaultSchema */
  ],
});

const Statuses = {
  loading: "Loading...",
  parsing: "Parsing",
  ready: "ready",
};

const editorTheme = {
  [ThemeNames.DARK]: "vs-dark",
  [ThemeNames.LIGHT]: "vs-light",
};

const EDITOR_ID = `workflow-yaml-editor`;

export const WorkflowEditor = (props) => {
  const {
    value,
    placeholder,
    disabled,
    onChange,
    editorRef = { current: {} },
  } = props;

  const clickedWord = useRef(null);
  const ref = useRef(null);
  const tabsRef = useRef(null);

  const filesRef = useRef();

  const [files, setFiles] = useState({
    wd: { uri: Uri.parse("wd"), error: false },
  });

  const [currentFile, setCurrentFile] = useState(files.wd.uri); // uri
  const [showPlaceholer, setShowPlaceholer] = useState(false);
  const [status, setStatus] = useState(Statuses.loading);

  const { theme } = useTheme();

  useEffect(() => {
    if (ref.current) return;
    monaco.editor.getModels().forEach((model) => model.dispose());
    window.gmonaco = monaco.editor;

    const editor = monaco.editor.create(document.getElementById(EDITOR_ID), {
      automaticLayout: true,
      model: monaco.editor.createModel(props.value, "yaml", files.wd.uri),
      theme: editorTheme[theme],
      dragAndDrop: false,
      quickSuggestions: {
        other: true,
        comments: false,
        strings: true,
      },
    });

    editorRef.current.editor = editor;
    editorRef.current.globalMonaco = monaco.editor;
    editorRef.current.save = mergeAndSave;
    editorRef.current.updateWithExtractWd = updateWithExtractWd;

    editor.onMouseDown(({ target: { position } }) => {
      const line = editor.getModel().getLineContent(position.lineNumber);
      clickedWord.current = getWordAt(line, position.column);
    });

    editor.onMouseUp(({ event, target: { position } }) => {
      const line = editor.getModel().getLineContent(position.lineNumber);
      const upWord = getWordAt(line, position.column);

      if (
        event.altKey &&
        upWord === clickedWord.current &&
        upWord.startsWith("$FILE_")
      ) {
        const fileKey = upWord.replace("$FILE_", "").toLowerCase();
        if (filesRef.current[fileKey]) {
          setCurrentFile(filesRef.current[fileKey]);
        }
      }
    });

    editor.onDidBlurEditorWidget(() => {
      if (editor.getValue() === "") {
        setShowPlaceholer(true);
      }
    });

    editor.onDidFocusEditorWidget(() => {
      setShowPlaceholer(false);
    });

    editor.onDidChangeModel((e) => {
      const model = editor.getModel();
      if (!model) return;
      setShowPlaceholer(model.getValue() === "");
    });

    editor.onDidChangeModelContent(() => {
      const model = editor.getModel();
      if (!model) return;
      setShowPlaceholer(model.getValue() === "");
    });

    // editor.onDid;
    monaco.editor.onDidChangeMarkers((uris) => {
      const updatedFiles = Object.entries(filesRef.current).reduce(
        (acc, [key, fileData]) => {
          acc[key] = {
            uri: fileData.uri,
            error: hasModelErrorsByResourse(fileData.uri),
          };
          return acc;
        },
        {},
      );

      setFiles(updatedFiles);
    });

    ref.current = editor;
    window.monaco = editor;

    return () => {
      const models = monaco.editor.getModels();
      models.forEach((model) => model.dispose());
      console.log(models);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createFilesByWd = useCallback(
    (wd) => {
      if (typeof wd !== "string" || !monaco) {
        return "";
      }

      const parsedFiles = {};
      const { workflow, ...base64entries } = parseWorkflowYaml(wd);
      for (const [key, value] of Object.entries(base64entries)) {
        const newFile = Uri.parse(key);
        parsedFiles[key] = { uri: newFile, error: false };

        if (monaco.editor.getModel(newFile)) {
          continue;
        }

        monaco.editor.createModel(value, "yaml", newFile);
      }

      setFiles({ wd: files.wd, ...parsedFiles });
      monaco.editor.getModel(files.wd.uri)?.setValue(workflow);
      return workflow;
    },
    [files],
  );

  useEffect(() => {
    filesRef.current = files;
    editorRef.current.files = files;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  const updateWithExtractWd = useCallback(function (value) {
    monaco.editor.getModel(files.wd.uri)?.setValue(value ?? "");
    extractBase64();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    monaco.editor.getModel(files.wd.uri)?.setValue(value ?? "");
    setStatus(Statuses.ready);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  useEffect(() => {
    if (currentFile) {
      const model = monaco.editor.getModel(currentFile);
      ref.current.setModel(model);
    }
  }, [currentFile]);

  useEffect(() => {
    tabsRef.current.onwheel = (evt) => {
      evt.preventDefault();
      tabsRef.current.scrollLeft += evt.deltaY;
    };
  }, []);

  useEffect(() => {
    const isDisabled = disabled || status !== Statuses.ready;

    const options = {
      readOnly: isDisabled,
      theme: editorTheme[theme],
    };

    ref.current?.updateOptions(options);
  }, [disabled, status, theme]);

  // function deleteFile(uri) {
  //   const updatedFiles = { ...files };

  //   // Update current file, to previos tab index
  //   if (currentFile === uri) {
  //     const arrayOfFiles = Object.entries(updatedFiles);
  //     const idx = arrayOfFiles.findIndex(([_, file]) => file.path === uri.path);
  //     const prevIdx = Math.max(idx - 1, 0);
  //     const prev = arrayOfFiles[prevIdx][1];
  //     setCurrentFile(prev);
  //   }

  //   delete updatedFiles[uri.path.slice(1)];

  //   monaco.editor.getModel(uri).dispose();
  //   setFiles(updatedFiles);
  // }

  function mergeAndSave({ onError, onSuccess }) {
    try {
      const { wd, ...otherFiles } = filesRef.current;

      const errorMarkers = getMonacoErrors();

      if (errorMarkers.length) {
        const files = [
          ...new Set(errorMarkers.map((marker) => marker.resource.path)),
        ].join("\n");

        throw new Error(t`Cannot save, workflow syntax error in ${files}`);
      }

      const wdModel = monaco.editor.getModel(files.wd.uri);
      let wdValue = wdModel.getValue();

      // Validating
      for (const [tabName, { uri }] of Object.entries(filesRef.current)) {
        if (uri?.path?.includes("requirements")) continue;
        if (isEmptyYaml(monaco.editor.getModel(uri).getValue())) {
          throw new Error(t`Cannot save, file ${tabName} is empty`);
        }
      }

      for (const [tabName, { uri }] of Object.entries(otherFiles)) {
        const tabModel = monaco.editor.getModel(uri);
        const tabValue = tabModel.getValue();
        wdValue = wdValue.replace(`$FILE_${tabName}`, encode(tabValue));
      }

      setCurrentFile(wd.uri);
      setFiles({ wd });

      wdModel.setValue(wdValue);
      onChange(wdValue);
      onSuccess(wdValue);
    } catch (e) {
      console.log(e);
      onError(e.message);
    }
  }

  function extractBase64() {
    const errors = getMonacoErrors();

    if (errors.length) {
      showToast(getErrorToast(t`Cannot parse yaml, while has syntax error`));
    } else {
      createFilesByWd(monaco.editor.getModel(files.wd.uri).getValue());
    }
  }

  return (
    <div>
      <div className="top-toobar">
        <div className="file-tabs-container" ref={tabsRef}>
          {Object.entries(files).map(([key, { uri, error }]) => {
            return (
              <div
                onClick={() => setCurrentFile(uri)}
                className={classNames("file-tab", {
                  active: currentFile?.path === uri.path,
                  error: Boolean(error),
                })}
                key={uri.path}
              >
                <span>{uri.path}</span>
                {/* {key !== "wd" && (
                  <Button
                    className="close-button"
                    type="button"
                    icon="cross"
                    minimal
                    small
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteFile(uri);
                    }}
                  />
                )} */}
              </div>
            );
          })}
        </div>
        <div className="actions">
          <Button
            icon="clean"
            intent={Intent.WARNING}
            outlined
            onClick={extractBase64}
          >
            <Trans>Extract base64</Trans>
          </Button>
        </div>
      </div>

      <div className={classNames("editor")} id={EDITOR_ID}>
        <div className="monaco-placeholder" hidden={!showPlaceholer}>
          {placeholder}
        </div>
      </div>
    </div>
  );
};
