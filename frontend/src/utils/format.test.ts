import { describe, expect, it } from "vitest";

import {
  displayStoredFileName,
  exportModelNamePart,
  formatFileSize,
  formatGeojsonSummary,
  formatObjectCount,
  formatRuntimeMinutes,
  formatTrainingResultDate,
  isValidExportModelName,
  runningProgressLabel,
} from "./format";

describe("format helpers", () => {
  it("normalizes Triton export model name parts", () => {
    expect(exportModelNamePart("Rivers Kanopus 0806")).toBe("rivers_kanopus_0806");
    expect(isValidExportModelName("rivers_kanopus_0806")).toBe(true);
    expect(isValidExportModelName("Реки")).toBe(false);
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
