import type { JsonObject } from "./datasetEditor";

type AnnotationFile = { name: string; size: number; text: () => Promise<string> };

export const NEW_SCENE_REVISION = "new";

export async function readAnnotationFiles(files: AnnotationFile[]) {
  if (!files.length || files.length > 100) throw new Error("Выберите от 1 до 100 файлов GeoJSON.");
  if (files.reduce((sum, file) => sum + file.size, 0) > 50 * 1024 * 1024) {
    throw new Error("Общий размер выбранных GeoJSON не должен превышать 50 МиБ.");
  }
  const names = new Set<string>();
  const scenes: { annotation_name: string; geojson: JsonObject }[] = [];
  for (const file of files) {
    const name = file.name.toLowerCase();
    if (!name.endsWith(".geojson") || name.endsWith("_footprint.geojson") || /[\\/]/.test(name)) {
      throw new Error(`${file.name}: нужен файл разметки .geojson, а не служебный footprint.`);
    }
    if (names.has(name)) throw new Error(`Выбраны файлы с одинаковым именем: ${file.name}`);
    names.add(name);
    let payload: unknown;
    try { payload = JSON.parse((await file.text()).replace(/^\uFEFF/, "")); }
    catch { throw new Error(`${file.name}: не удалось прочитать JSON. Проверьте формат и кодировку UTF-8.`); }
    if (!payload || typeof payload !== "object" || Array.isArray(payload)
      || (payload as JsonObject).type !== "FeatureCollection" || !Array.isArray((payload as JsonObject).features)) {
      throw new Error(`${file.name}: ожидается FeatureCollection со списком features.`);
    }
    scenes.push({ annotation_name: file.name, geojson: payload as JsonObject });
  }
  return scenes;
}
