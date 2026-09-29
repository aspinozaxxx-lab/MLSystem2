import { describe, expect, it } from "vitest";
import { readAnnotationFiles } from "./datasetEditorImport";

const payload = JSON.stringify({ type: "FeatureCollection", features: [] });
const file = (name: string, text = payload, size = text.length) => ({ name, size, text: async () => text });

describe("загрузка GeoJSON в черновики", () => {
  it("читает несколько файлов UTF-8, включая BOM и пустую разметку", async () => {
    const scenes = await readAnnotationFiles([file("папка_один.geojson", `\uFEFF${payload}`), file("папка_два.geojson")]);
    expect(scenes.map(scene => scene.annotation_name)).toEqual(["папка_один.geojson", "папка_два.geojson"]);
    expect(scenes.every(scene => Array.isArray(scene.geojson.features))).toBe(true);
  });
  it("отклоняет весь пакет при ошибке второго файла", async () => {
    await expect(readAnnotationFiles([file("папка_один.geojson"), file("папка_два.geojson", "{не JSON}")])).rejects.toThrow("папка_два.geojson");
    await expect(readAnnotationFiles([file("папка_один.geojson", "[]")])).rejects.toThrow("FeatureCollection");
  });
  it("не принимает повторяющиеся имена, footprint и архивы", async () => {
    await expect(readAnnotationFiles([file("папка_один.geojson"), file("ПАПКА_ОДИН.GEOJSON")])).rejects.toThrow("одинаковым именем");
    await expect(readAnnotationFiles([file("папка_один_footprint.geojson")])).rejects.toThrow("footprint");
    await expect(readAnnotationFiles([file("датасет.zip")])).rejects.toThrow(".geojson");
  });
  it("проверяет размер и количество перед чтением файлов", async () => {
    await expect(readAnnotationFiles([file("папка_один.geojson", payload, 50 * 1024 * 1024 + 1)])).rejects.toThrow("50 МиБ");
    await expect(readAnnotationFiles(Array.from({ length: 101 }, (_, i) => file(`папка_${i}.geojson`)))).rejects.toThrow("100");
  });
});
