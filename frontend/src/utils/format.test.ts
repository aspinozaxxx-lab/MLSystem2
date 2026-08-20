import { describe, expect, it } from "vitest";

import {
  defaultTrainingZipModelName,
  displayStoredFileName,
  exportModelNamePart,
  formatFileSize,
  formatGeojsonSummary,
  formatObjectCount,
  formatRuntimeMinutes,
  formatTrainingResultDate,
  imageryTypeForInputChannels,
  isPrimaryDataset,
  isValidExportModelName,
  runningProgressLabel,
} from "./format";

describe("format helpers", () => {
  it("normalizes Triton export model name parts", () => {
    expect(exportModelNamePart("Rivers Kanopus 0806")).toBe("rivers_kanopus_0806");
    expect(isValidExportModelName("rivers_kanopus_0806")).toBe(true);
    expect(isValidExportModelName("Реки")).toBe(false);
  });

  it("предлагает техническое имя класса с суффиксом типа снимков", () => {
    expect(defaultTrainingZipModelName(
      { dataset_key: "deforestation-id" },
      [{
        key: "deforestation-id",
        class_technical_name: "forest_cuttings",
        model_name_stem: "deforestation",
        imagery_type: "kanopus",
      }],
    )).toBe("forest_cuttings_kanopus");
    expect(defaultTrainingZipModelName(
      { dataset_key: "zu-id" },
      [{
        key: "zu-id",
        model_name_stem: "zu500",
        imagery_type: "ortho",
      }],
    )).toBe("zu500_orto");
  });

  it("сохраняет совместимый fallback для legacy annotation_file", () => {
    expect(defaultTrainingZipModelName(
      { dataset_key: "rivers" },
      [{
        key: "rivers",
        annotation_file: "/data/MLMarkup/Реки/main/rivers.geojson",
        imagery_type: "kanopus",
      }],
    )).toBe("rivers_kanopus");
  });

  it("учитывает явный признак основного датасета", () => {
    expect(isPrimaryDataset({ is_primary: false })).toBe(false);
    expect(isPrimaryDataset({ is_primary: true })).toBe(true);
    expect(isPrimaryDataset({})).toBe(false);
  });

  it("определяет тип снимков по числу входных каналов", () => {
    expect(imageryTypeForInputChannels(4)).toBe("kanopus");
    expect(imageryTypeForInputChannels(3)).toBe("ortho");
    expect(imageryTypeForInputChannels(2)).toBeNull();
  });

  it("formats file sizes and running progress", () => {
    expect(formatFileSize(1536)).toBe("1.5 KB");
    expect(formatRuntimeMinutes(30)).toBe("30м");
    expect(formatRuntimeMinutes(90)).toBe("1:30");
    expect(runningProgressLabel("inference", { current: 3, total: 10, elapsed_minutes: 2 })).toBe("3/10, 2 мин");
  });

  it("formats GeoJSON object summaries", () => {
    expect(formatObjectCount(1)).toBe("1 объект");
    expect(formatObjectCount(2)).toBe("2 объекта");
    expect(formatObjectCount(5)).toBe("5 объектов");
    expect(formatObjectCount(11)).toBe("11 объектов");
    expect(formatObjectCount(null)).toBe("— объектов");
    expect(formatGeojsonSummary(500, 3.5 * 1024 * 1024)).toBe("500 объектов - 3.5 MB");
  });

  it("formats training result dates by status", () => {
    expect(
      formatTrainingResultDate(
        "ok",
        "2026-06-10T12:30:00",
        "2026-06-10T12:00:00",
        "2026-06-10T11:00:00",
      ),
    ).toContain("12:30");
    expect(
      formatTrainingResultDate(
        "error",
        null,
        "2026-06-10T12:00:00",
        "2026-06-10T11:00:00",
      ),
    ).toContain("12:00");
    expect(formatTrainingResultDate("error", null, null, "2026-06-10T11:00:00")).toContain("11:00");
  });

  it("normalizes displayed stored file names", () => {
    expect(displayStoredFileName("Засоления\\main_segformer b2_07_38_06_06.geojson")).toBe("Засоления_main_segformer b2_07_38_06_06.geojson");
  });
});
