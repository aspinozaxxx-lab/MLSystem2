import { describe, expect, it } from "vitest";

import { exportModelNamePart, formatFileSize, isValidExportModelName, runningProgressLabel } from "./format";

describe("format helpers", () => {
  it("normalizes Triton export model name parts", () => {
    expect(exportModelNamePart("Rivers Kanopus 0806")).toBe("rivers_kanopus_0806");
    expect(isValidExportModelName("rivers_kanopus_0806")).toBe(true);
    expect(isValidExportModelName("Реки")).toBe(false);
  });

  it("formats file sizes and running progress", () => {
    expect(formatFileSize(1536)).toBe("1.5 KB");
    expect(runningProgressLabel("inference", { current: 3, total: 10, elapsed_minutes: 2 })).toBe("3/10, 2 мин");
  });
});
