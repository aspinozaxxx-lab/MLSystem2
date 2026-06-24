import type { RuntimeProgress } from "../api/types";

export function formatDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${date.getFullYear()} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return `${pad(date.getDate())}.${pad(date.getMonth() + 1)}.${date.getFullYear()}`;
}

export function formatF1Score(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(3) : "—";
}

export function formatFileSize(value: number | null | undefined): string {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${formatDecimal(bytes / 1024)} KB`;
  return `${formatDecimal(bytes / 1024 / 1024)} MB`;
}

export function formatDecimal(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

export function integerOrNull(value: unknown): number | null {
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
}

export function shortVersion(version: string | null | undefined): string {
  if (!version) return "";
  return version.length > 10 ? version.slice(0, 10) : version;
}

export function exportModelNamePart(value: unknown): string {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "");
}

export function isValidExportModelName(value: string): boolean {
  return /^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$/.test(value);
}

export function runningProgressLabel(type: string | null | undefined, progress: RuntimeProgress | null | undefined): string {
  if (!progress) return type === "inference" ? "Инференс выполняется" : "Выполняется";
  const current = integerOrNull(progress.current);
  const total = integerOrNull(progress.total);
  const elapsed = integerOrNull(progress.elapsed_minutes);
  const pieces: string[] = [];
  if (current !== null && total !== null && total > 0) {
    pieces.push(`${current}/${total}`);
  }
  if (elapsed !== null) {
    pieces.push(`${elapsed} мин`);
  }
  return pieces.length ? pieces.join(", ") : "Выполняется";
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}
