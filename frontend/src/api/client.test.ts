import { afterEach, describe, expect, it, vi } from "vitest";

import { apiJson, downloadFilename } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("ошибки API", () => {
  it("передаёт код предложения обрезки вместе с сообщением сервера", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: "Разметка выходит за footprint.", code: "annotation_outside_footprint",
    }), { status: 400 })));
    await expect(apiJson("/dataset-editor/datasets/test/drafts/import")).rejects.toMatchObject({
      message: "Разметка выходит за footprint.", status: 400, code: "annotation_outside_footprint",
    });
  });
  it("сохраняет обычные сообщения и HTTP-ошибки без JSON", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Неверная роль." }), { status: 400 }))
      .mockResolvedValueOnce(new Response("Недоступно", { status: 503 }));
    vi.stubGlobal("fetch", fetcher);
    await expect(apiJson("/test")).rejects.toMatchObject({ message: "Неверная роль.", status: 400, code: undefined });
    await expect(apiJson("/test")).rejects.toMatchObject({ message: "HTTP 503", status: 503, code: undefined });
  });
});


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
