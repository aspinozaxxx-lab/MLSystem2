import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { installPseudoModuleRecovery } from "./moduleRecovery";

let stop: () => void;
let target: EventTarget;
let page: { hash: string; reload: ReturnType<typeof vi.fn> };
let values: Map<string, string>;
function fail() { const event = new Event("vite:preloadError", { cancelable: true }); target.dispatchEvent(event); return event; }

beforeEach(() => {
  target = new EventTarget();
  page = { hash: "#/pseudo-markup/result", reload: vi.fn() };
  values = new Map();
  vi.stubGlobal("window", target);
  vi.stubGlobal("location", page);
  vi.stubGlobal("navigator", { onLine: true });
  vi.stubGlobal("sessionStorage", { getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value) });
  stop = installPseudoModuleRecovery();
});
afterEach(() => { stop(); vi.unstubAllGlobals(); });

describe("Обновление старой вкладки просмотра", () => {
  it("перечитывает просмотр при пропаже старого модуля и не зацикливается", () => {
    expect(fail().defaultPrevented).toBe(true);
    expect(page.reload).toHaveBeenCalledOnce();
    expect(fail().defaultPrevented).toBe(false);
    expect(page.reload).toHaveBeenCalledOnce();
  });
  it.each(["#/dataset-editor", "#/start", "#/templates"])("не теряет изменения в форме %s", (hash) => {
    page.hash = hash;
    expect(fail().defaultPrevented).toBe(false);
    expect(page.reload).not.toHaveBeenCalled();
  });
  it("не перезагружает страницу без сети или доступного хранилища защиты от цикла", () => {
    vi.stubGlobal("navigator", { onLine: false });
    fail();
    vi.stubGlobal("navigator", { onLine: true });
    vi.stubGlobal("sessionStorage", { getItem: () => { throw new Error("Хранилище недоступно"); } });
    fail();
    expect(page.reload).not.toHaveBeenCalled();
  });
});
