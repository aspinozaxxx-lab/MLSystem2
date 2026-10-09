import { afterEach, describe, expect, it, vi } from "vitest";

import { startMetrica, usagePageForRoute } from "./usage";

afterEach(() => vi.unstubAllGlobals());

function browser() {
  const target = { ym: vi.fn(), location: { origin: "https://grovika.ru" } };
  const head = { appendChild: vi.fn() };
  vi.stubGlobal("window", target);
  vi.stubGlobal("document", {
    querySelector: vi.fn().mockReturnValue(null),
    createElement: vi.fn().mockReturnValue({ dataset: {} }), head,
  });
  return { target, head };
}

const config = { metrica_counter_id: 113578839, metrica_user_id: "user-one" };

describe("Яндекс Метрика Гровики", () => {
  it("включает Вебвизор и явные просмотры без ключей датасета в адресах", () => {
    const { target, head } = browser();
    const tracker = startMetrica(config)!;
    const page = usagePageForRoute(["results", "ОКС500/личный_датасет?token=private"]);
    tracker.page(page);
    expect(head.appendChild.mock.calls[0][0].src).toBe("https://mc.yandex.ru/metrika/tag.js?id=113578839");
    expect(target.ym).toHaveBeenCalledWith(config.metrica_counter_id, "init", expect.objectContaining({ defer: true, webvisor: true }));
    expect(target.ym).toHaveBeenCalledWith(config.metrica_counter_id, "setUserID", "user-one");
    const hit = target.ym.mock.calls.find((args) => args[1] === "hit")!;
    expect(hit[2]).toBe("https://grovika.ru/#/usage/dataset_results");
    expect(JSON.stringify(target.ym.mock.calls)).not.toContain("private");
    expect(hit[3].params.grovika.pages["user-one"]).toBe("Результаты датасета");
  });

  it("использует загруженный обработчик вместо старой очереди", () => {
    const { target } = browser();
    const tracker = startMetrica(config)!;
    target.ym = vi.fn();
    tracker.action("training_start", 422, "user-one");
    expect(target.ym).toHaveBeenCalledWith(config.metrica_counter_id, "params", {
      grovika: { features: { "user-one": { "Запуск обучения": "Ошибка" } } },
    });
    expect(target.ym.mock.calls.some((args) => args[1] === "reachGoal")).toBe(false);
    tracker.action("training_start", 200, "user-one");
    expect(target.ym).toHaveBeenCalledWith(config.metrica_counter_id, "reachGoal", "grovika_training_start");
    expect(target.ym.mock.calls.filter((args) => args[1] === "params")).toHaveLength(2);
  });

  it("исключает поздний ответ другого аккаунта и прекращает отправку после выхода", () => {
    const { target } = browser();
    const tracker = startMetrica(config)!;
    target.ym.mockClear();
    tracker.action("training_start", 200, "previous-user");
    tracker.action("unknown", 200, "user-one");
    expect(target.ym).not.toHaveBeenCalled();
    tracker.stop();
    target.ym.mockClear();
    tracker.action("training_start", 200, "user-one");
    tracker.page("home");
    expect(target.ym).not.toHaveBeenCalled();
  });

  it("при блокировке скрипта удаляет очередь вышедшего пользователя", () => {
    const { target } = browser();
    const queued = Object.assign(vi.fn(), { a: [[config.metrica_counter_id, "hit", "old-view"], [42, "hit", "other-counter"]] });
    vi.stubGlobal("window", { ...target, ym: queued });
    startMetrica(config)!.stop();
    expect(queued.a).toEqual([[42, "hit", "other-counter"]]);
  });

  it("не загружает отключённый счётчик и переносит ошибки библиотеки", () => {
    const { target, head } = browser();
    expect(startMetrica({ ...config, metrica_counter_id: null })).toBeNull();
    expect(head.appendChild).not.toHaveBeenCalled();
    target.ym.mockImplementation(() => { throw new Error("Счётчик недоступен"); });
    expect(() => startMetrica(config)!.page("home")).not.toThrow();
  });

  it("разделяет просмотр псевды, сравнение, тестовый F1 и оба вида шаблонов", () => {
    expect(usagePageForRoute(["pseudo-markup", "compare", "ids"])).toBe("pseudo_compare");
    expect(usagePageForRoute(["pseudo-markup", "result-id"])).toBe("pseudo_view");
    expect(usagePageForRoute(["test-f1", "result-id"])).toBe("test_f1");
    expect(usagePageForRoute(["templates", "inference"])).toBe("inference_templates");
    expect(usagePageForRoute(["templates"])).toBe("training_templates");
  });
});
