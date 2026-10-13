import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const classes = [{ key: "озёра" }, { key: "зу500" }];
let storage: Map<string, string>;

beforeEach(() => {
  vi.resetModules();
  storage = new Map();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
  });
});
afterEach(() => vi.unstubAllGlobals());

async function page() {
  const { useRememberedClass } = await import("./ClassNavigation");
  let choose: (key: string) => void;
  function Probe({ available }: { available: typeof classes | null }) {
    const selection = useRememberedClass(available);
    choose = selection.setClassKey;
    return <output>{selection.classKey}</output>;
  }
  return {
    render: (available = classes as typeof classes | null) => renderToStaticMarkup(<Probe available={available} />),
    choose: (key: string) => choose(key),
  };
}

describe("общий выбор класса", () => {
  it("переносит выбор между страницами, включая обратный переход и перезагрузку", async () => {
    const results = await page();
    results.render();
    results.choose("зу500");
    const markups = await page();
    expect(markups.render()).toContain("зу500");
    const editor = await page();
    expect(editor.render([...classes].reverse())).toContain("зу500");
    editor.choose("озёра");
    expect(results.render()).toContain("озёра");
    expect(storage.get("grovika:selected-class")).toBe("озёра");
    vi.resetModules();
    expect((await page()).render()).toContain("озёра");
  });

  it("не теряет выбор при загрузке, пустом или ограниченном каталоге", async () => {
    storage.set("grovika:selected-class", "зу500");
    const results = await page();
    expect(results.render(null)).toContain("зу500");
    expect(results.render([{ key: "озёра" }])).toContain("озёра");
    expect(results.render([])).toBe("<output></output>");
    expect((await page()).render()).toContain("зу500");
    expect(storage.get("grovika:selected-class")).toBe("зу500");
  });

  it("подхватывает прежний выбор и продолжает работать без хранилища браузера", async () => {
    storage.set("grovika:results:class", "зу500");
    const results = await page();
    expect(results.render()).toContain("зу500");
    vi.stubGlobal("localStorage", {
      getItem: () => { throw new Error("Хранилище недоступно"); },
      setItem: () => { throw new Error("Хранилище недоступно"); },
    });
    results.choose("озёра");
    expect((await page()).render()).toContain("озёра");
  });
});
