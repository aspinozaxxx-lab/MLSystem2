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
};

export async function apiJson<T>(path: string, options: JsonOptions = {}): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: options.method || "GET",
    credentials: "same-origin",
    headers: options.body === undefined ? undefined : { "Content-Type": "application/json" },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  });
  if (response.status === 401 && options.authOptional) {
    return null as T;
  }
  if (!response.ok) {
    throw await responseError(response);
  }
  if (response.status === 204) {
    return null as T;
  }
  return (await response.json()) as T;
}

export async function apiForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  return (await response.json()) as T;
}

export async function apiDownload(path: string, form: FormData): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(apiUrl(path), {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
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
