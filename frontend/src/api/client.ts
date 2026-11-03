const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type JsonOptions = {
  method?: string;
  body?: unknown;
  authOptional?: boolean;
  signal?: AbortSignal;
  onUploadProgress?: (progress: UploadProgress) => void;
};

export type UploadProgress = { loaded: number; total: number | null; complete: boolean };

export async function apiJson<T>(path: string, options: JsonOptions = {}): Promise<T> {
  const init = {
    method: options.method || "GET",
    credentials: "same-origin",
    headers: options.body === undefined ? undefined : { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  } satisfies RequestInit;
  const response = options.onUploadProgress
    ? await uploadResponse(path, init, options.onUploadProgress)
    : await fetch(apiUrl(path), init);
  if (response.status === 401 && options.authOptional) {
    return null as T;
  }
  usageAction(response);
  if (!response.ok) {
    throw await responseError(response);
  }
  if (response.status === 204) {
    return null as T;
  }
  return (await response.json()) as T;
}

export async function apiForm<T>(path: string, form: FormData, options: Pick<JsonOptions, "onUploadProgress" | "signal"> = {}): Promise<T> {
  const init = {
    method: "POST",
    credentials: "same-origin",
    body: form,
    signal: options.signal,
  } satisfies RequestInit;
  const response = options.onUploadProgress
    ? await uploadResponse(path, init, options.onUploadProgress)
    : await fetch(apiUrl(path), init);
  usageAction(response);
  if (!response.ok) {
    throw await responseError(response);
  }
  return (await response.json()) as T;
}

function uploadResponse(path: string, init: RequestInit, onProgress: (progress: UploadProgress) => void): Promise<Response> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let progress: UploadProgress = { loaded: 0, total: null, complete: false };
    const abort = () => xhr.abort();
    const cleanup = () => init.signal?.removeEventListener("abort", abort);
    const fail = (error: Error) => { cleanup(); reject(error); };
    xhr.upload.addEventListener("progress", event => {
      progress = { loaded: event.loaded, total: event.lengthComputable ? event.total : null, complete: false };
      onProgress(progress);
    });
    xhr.upload.addEventListener("load", () => {
      progress = { ...progress, loaded: progress.total ?? progress.loaded, complete: true };
      onProgress(progress);
    });
    xhr.addEventListener("load", () => {
      cleanup();
      if (!xhr.status) { reject(new Error("Не удалось получить ответ сервера.")); return; }
      try {
        const headers = new Headers();
        xhr.getAllResponseHeaders().trim().split(/[\r\n]+/).forEach(line => {
          const colon = line.indexOf(":");
          if (colon > 0) headers.append(line.slice(0, colon).trim(), line.slice(colon + 1).trim());
        });
        resolve(new Response(xhr.responseText || null, { status: xhr.status, headers }));
      } catch (error) { reject(error); }
    });
    xhr.addEventListener("error", () => fail(new Error("Не удалось передать файлы на сервер. Проверьте соединение и повторите загрузку.")));
    xhr.addEventListener("timeout", () => fail(new Error("Сервер не ответил вовремя. Повторите загрузку.")));
    xhr.addEventListener("abort", () => fail(new DOMException("Загрузка отменена", "AbortError")));
    xhr.open(init.method || "POST", apiUrl(path));
    xhr.withCredentials = true;
    new Headers(init.headers).forEach((value, key) => xhr.setRequestHeader(key, value));
    if (init.signal?.aborted) { fail(new DOMException("Загрузка отменена", "AbortError")); return; }
    init.signal?.addEventListener("abort", abort, { once: true });
    xhr.send(init.body as XMLHttpRequestBodyInit | null | undefined);
  });
}

export async function apiDownload(path: string, form: FormData): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  usageAction(response);
  if (!response.ok) {
    throw await responseError(response);
  }
  return {
    blob: await response.blob(),
    filename: downloadFilename(response),
  };
}

export async function apiDownloadJson(path: string, body: unknown): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  usageAction(response);
  if (!response.ok) {
    throw await responseError(response);
  }
  return {
    blob: await response.blob(),
    filename: downloadFilename(response),
  };
}

export async function apiDownloadGet(path: string): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(apiUrl(path), {
    method: "GET",
    credentials: "same-origin",
  });
  usageAction(response);
  if (!response.ok) {
    throw await responseError(response);
  }
  return {
    blob: await response.blob(),
    filename: downloadFilename(response),
  };
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function downloadFilename(response: Response): string {
  const header = response.headers.get("content-disposition") || "";
  const utfMatch = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utfMatch) {
    try {
      return decodeURIComponent(utfMatch[1]);
    } catch {
      return utfMatch[1];
    }
  }
  const match = header.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : "";
}

function apiUrl(path: string): string {
  return path === API_PREFIX || path.startsWith(`${API_PREFIX}/`) ? path : `${API_PREFIX}${path}`;
}

function usageAction(response: Response): void {
  const name = response.headers.get("X-Grovika-Action");
  if (name && typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("grovika:action", {
      detail: { name, status: response.status, userId: response.headers.get("X-Grovika-User") },
    }));
  }
}

async function responseError(response: Response): Promise<ApiError> {
  try {
    const payload = (await response.json()) as { detail?: unknown; code?: unknown };
    if (typeof payload.detail === "string") {
      return new ApiError(payload.detail, response.status, typeof payload.code === "string" ? payload.code : undefined);
    }
  } catch {
    // Ответ без читаемого JSON сохраняет обычное сообщение с HTTP-статусом.
  }
  return new ApiError(`HTTP ${response.status}`, response.status);
}
