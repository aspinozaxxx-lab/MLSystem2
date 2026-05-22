const g = require("generate-template-files");

g.generateTemplateFiles([
  {
    option: "one-file-component",
    defaultCase: "(pascalCase)",
    entry: {
      folderPath: "./tools/templates/component/index.js"
    },
    stringReplacers: ["__name__"],
    output: {
      path: "./src/__name__(kebabCase).js",
      pathAndFileNameDefaultCase: "(kebabCase)"
    },
    onComplete: results => console.log("results", results)
  },
  {
    option: "component",
    defaultCase: "(pascalCase)",
    entry: {
      folderPath: "./tools/templates/component"
    },
    stringReplacers: ["__name__"],
    output: {
      path: "./src/components/__name__(kebabCase)",
      pathAndFileNameDefaultCase: "(kebabCase)"
    },
    onComplete: results => console.log("results", results)
  },
  {
    option: "page",
    defaultCase: "(pascalCase)",
    entry: {
      folderPath: "./tools/templates/container"
    },
    stringReplacers: ["__name__"],
    output: {
      path: "./src/pages/__name__(kebabCase)",
      pathAndFileNameDefaultCase: "(kebabCase)"
    },
    onComplete: results => console.log("results", results)
  },
  {
    option: "container",
    defaultCase: "(pascalCase)",
    entry: {
      folderPath: "./tools/templates/container"
    },
    stringReplacers: ["__name__"],
    output: {
      path: "./src/containers/__name__(kebabCase)",
      pathAndFileNameDefaultCase: "(kebabCase)"
    },
    onComplete: results => console.log("results", results)
  }
]);
