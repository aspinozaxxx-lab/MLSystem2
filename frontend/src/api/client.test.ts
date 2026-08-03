import { describe, expect, it } from "vitest";

import { downloadFilename } from "./client";


describe("downloadFilename", () => {
  it("decodes a UTF-8 filename from Content-Disposition", () => {
    const response = new Response(null, {
      headers: {
        "Content-Disposition":
          "attachment; filename=\"scene-list.txt\"; filename*=UTF-8''%D0%A0%D0%B0%D0%B7%D0%BC%D0%B5%D1%82%D0%BA%D0%B0%20%D1%80%D0%B5%D0%BA.txt",
      },
    });

    expect(downloadFilename(response)).toBe("Разметка рек.txt");
  });
});
