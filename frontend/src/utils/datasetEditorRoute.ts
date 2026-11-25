/** Адрес редактора хранит ключ датасета и имя его поснимочной разметки. */
export function datasetEditorHash(datasetKey: string, annotationName = ""): string {
  const dataset = "#/dataset-editor/" + encodeURIComponent(datasetKey);
  return annotationName ? dataset + "/" + encodeURIComponent(annotationName) : dataset;
}

function decodeSegment(segment?: string): string {
  try { return decodeURIComponent(segment || ""); }
  catch { return segment || ""; }
}

export function datasetEditorRoute(route: readonly string[]) {
  if (route[0] !== "dataset-editor") return undefined;
  return { datasetKey: decodeSegment(route[1]), annotationName: decodeSegment(route[2]) };
}
