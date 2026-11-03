import { afterEach, describe, expect, it, vi } from "vitest";

import { apiForm, apiJson, downloadFilename } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("ошибки API", () => {
  it("передаёт действие с ID серверной сессии и не считает обычное чтение", async () => {
    const events = new EventTarget();
    const listener = vi.fn();
    events.addEventListener("grovika:action", listener);
    vi.stubGlobal("window", events);
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(new Response("{}", { headers: { "X-Grovika-Action": "training_start", "X-Grovika-User": "one" } }))
      .mockResolvedValueOnce(new Response("{}")));
    await apiJson("/training-jobs", { method: "POST", body: {} });
    await apiJson("/queues/count");
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].detail).toEqual({ name: "training_start", status: 200, userId: "one" });
  });
  it("передаёт отмену запроса серверу без замены AbortError на ошибку API", async () => {
    const controller = new AbortController();
    const aborted = new DOMException("Запрос отменён", "AbortError");
    const fetcher = vi.fn().mockRejectedValue(aborted);
    vi.stubGlobal("fetch", fetcher);
    await expect(apiJson("/results/pseudo-markup/compare/counts", { signal: controller.signal })).rejects.toBe(aborted);
    expect(fetcher.mock.calls[0][1].signal).toBe(controller.signal);
  });
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

function uploadFixture() {
  const requests: FakeRequest[] = [];
  class FakeRequest extends EventTarget {
    upload = new EventTarget();
    status = 0;
    responseText = "";
    responseHeaders = "";
    withCredentials = false;
    open = vi.fn();
    send = vi.fn();
    setRequestHeader = vi.fn();
    abort = vi.fn(() => this.dispatchEvent(new Event("abort")));
    getAllResponseHeaders() { return this.responseHeaders; }
    constructor() { super(); requests.push(this); }
    progress(loaded: number, total = 0) {
      this.upload.dispatchEvent(Object.assign(new Event("progress"), { loaded, total, lengthComputable: total > 0 }));
    }
    respond(status: number, payload: unknown, headers = "") {
      this.status = status; this.responseText = JSON.stringify(payload); this.responseHeaders = headers;
      this.dispatchEvent(new Event("load"));
    }
  }
  vi.stubGlobal("XMLHttpRequest", FakeRequest);
  return requests;
}

describe("передача файлов с прогрессом", () => {
  it("показывает фактические байты и завершение передачи отдельно от ответа API", async () => {
    const requests = uploadFixture(), progress = vi.fn(), form = new FormData();
    form.append("archive", new Blob(["архив"]), "выборка.zip");
    const promise = apiForm("/test-samples/набор/editor/import-archive", form, { onUploadProgress: progress });
    const settled = vi.fn(); void promise.then(settled);
    const xhr = requests[0];
    expect(xhr.open).toHaveBeenCalledWith("POST", "/api/v1/test-samples/набор/editor/import-archive");
    expect(xhr.withCredentials).toBe(true);
    expect(xhr.send).toHaveBeenCalledWith(form);
    expect(xhr.setRequestHeader).not.toHaveBeenCalled();
    xhr.progress(25, 100);
    expect(progress).toHaveBeenLastCalledWith({ loaded: 25, total: 100, complete: false });
    xhr.upload.dispatchEvent(new Event("load"));
    expect(progress).toHaveBeenLastCalledWith({ loaded: 100, total: 100, complete: true });
    await Promise.resolve(); expect(settled).not.toHaveBeenCalled();
    xhr.respond(200, { saved: true });
    await expect(promise).resolves.toEqual({ saved: true });
  });

  it("не придумывает процент, если длина передачи неизвестна", async () => {
    const requests = uploadFixture(), progress = vi.fn();
    const promise = apiForm("/upload", new FormData(), { onUploadProgress: progress });
    requests[0].progress(1024);
    expect(progress).toHaveBeenLastCalledWith({ loaded: 1024, total: null, complete: false });
    requests[0].respond(200, {}); await promise;
  });

  it("передаёт GeoJSON тем же транспортом, сохраняет заголовки действий и коды ошибок", async () => {
    const requests = uploadFixture(), progress = vi.fn(), events = new EventTarget(), action = vi.fn();
    events.addEventListener("grovika:action", action); vi.stubGlobal("window", events);
    const body = { scenes: [{ annotation_name: "контур.geojson" }] };
    const promise = apiJson("/test-samples/набор/editor/drafts/import", { method: "POST", body, onUploadProgress: progress });
    const xhr = requests[0];
    expect(xhr.send).toHaveBeenCalledWith(JSON.stringify(body));
    expect(xhr.setRequestHeader).toHaveBeenCalledWith("content-type", "application/json");
    xhr.respond(400, { detail: "Контур выходит за снимок", code: "annotation_outside_footprint" },
      "x-grovika-action: test_import\r\nx-grovika-user: session\r\n");
    await expect(promise).rejects.toMatchObject({ status: 400, code: "annotation_outside_footprint", message: "Контур выходит за снимок" });
    expect(action).toHaveBeenCalledOnce();
  });

  it("завершает ожидание при сетевой ошибке, отмене и истечении времени", async () => {
    const requests = uploadFixture();
    const failed = apiForm("/upload", new FormData(), { onUploadProgress: vi.fn() });
    requests[0].dispatchEvent(new Event("error")); await expect(failed).rejects.toThrow("Проверьте соединение");
    const controller = new AbortController();
    const aborted = apiForm("/upload", new FormData(), { onUploadProgress: vi.fn(), signal: controller.signal });
    controller.abort(); await expect(aborted).rejects.toMatchObject({ name: "AbortError" });
    expect(requests[1].abort).toHaveBeenCalledOnce();
    const timedOut = apiForm("/upload", new FormData(), { onUploadProgress: vi.fn() });
    requests[2].dispatchEvent(new Event("timeout")); await expect(timedOut).rejects.toThrow("Сервер не ответил");
  });
});
