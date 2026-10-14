import type { FormEvent } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PseudoMarkupForm } from "./App";
import { ApiError } from "./api/client";
import type { DatasetInfo, TrainingResultInfo } from "./api/types";
import { newsArticles } from "./News";

const formState = vi.hoisted(() => ({ updates: [] as unknown[] }));
vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return { ...actual, useState(initialValue: unknown) {
    const [value, setValue] = actual.useState(initialValue);
    return [value, (nextValue: unknown) => { formState.updates.push(nextValue); setValue(nextValue); }];
  } };
});

const result: TrainingResultInfo = {
  id: "сеть-28", model_name: "SegFormer B2", architecture: "smp_segformer_b2", input_channels: 3,
  epoch: 7, created_at: "2026-10-14T12:00:00Z", source: "manual", status: "ok", pipeline_variant: "object_f1_v2",
  quality_metric: "objects", validation_fold: 0, is_primary: false, can_continue_training: true, task: "binary",
};
const NativeFormData = FormData;
const trainingDataset: DatasetInfo = {
  key: "зу\\main", name: "ЗУ500\\main", is_custom: false, quality_metric: "objects", imagery_type: "ortho",
  source_available: true, is_primary: true, task: "binary", combined: false, managed: false,
  source_status: "current", hard_negative_count: 0, materialization_status: "not_applicable",
};
const makeProps = (overrides: Partial<Parameters<typeof PseudoMarkupForm>[0]> = {}) => ({
  datasetKey: "зу\\main", result, datasets: [trainingDataset], imageFolders: [], inferenceAvailable: true,
  run: async <T,>(operation: () => Promise<T>) => operation(), closeModal: vi.fn(), reload: vi.fn(async () => {}), ...overrides,
});

async function submit(props: Parameters<typeof PseudoMarkupForm>[0], values: Record<string, string | File>) {
  const currentTarget = {} as HTMLFormElement;
  vi.stubGlobal("FormData", class extends NativeFormData {
    constructor(form?: HTMLFormElement) {
      super();
      if (form === currentTarget) Object.entries(values).forEach(([key, value]) => this.set(key, value));
    }
  });
  let submitHandler: ((event: FormEvent<HTMLFormElement>) => Promise<void>) | undefined;
  function CaptureForm() {
    const form = PseudoMarkupForm(props);
    submitHandler = form.props.onSubmit;
    return form;
  }
  renderToStaticMarkup(<CaptureForm />);
  await submitHandler!({ preventDefault: vi.fn(), currentTarget } as unknown as FormEvent<HTMLFormElement>);
}

afterEach(() => { vi.unstubAllGlobals(); formState.updates = []; });

describe("чекпойнт ручной псевдоразметки", () => {
  it("показывает сеть, лучший чекпойнт по умолчанию и доступный выбор последнего", () => {
    const html = renderToStaticMarkup(<PseudoMarkupForm {...makeProps()} />);
    expect(html).toContain("Сеть: SegFormer B2");
    expect(html).toContain("Обучена на: ЗУ500\\main");
    expect(html).toContain("Чекпойнт для псевдоразметки");
    expect(html).toContain('name="checkpoint" checked="" value="best"');
    expect(html).toContain('name="checkpoint" value="last"');
    expect(html).toContain("Лучший · эпоха 7");
    expect(html).toContain("другие веса не подставляются");
  });

  it("не предлагает последний чекпойнт импортированной сети", () => {
    const html = renderToStaticMarkup(<PseudoMarkupForm {...makeProps({ result: { ...result, architecture: "external_torchscript" } })} />);
    expect(html).toContain('disabled="" name="checkpoint" value="last"');
    expect(html).toContain("У импортированной сети доступны только сохранённые веса.");
  });

  it.each(["best", "last"])("отправляет именно выбранный чекпойнт %s через существующий API", async (checkpoint) => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: "задание" })));
    vi.stubGlobal("fetch", fetcher);
    const props = makeProps();
    await submit(props, { dataset_key: "снимки", checkpoint });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0][0]).toBe("/api/v1/results/datasets/%D0%B7%D1%83%5Cmain/pseudo-markup");
    const request = fetcher.mock.calls[0][1];
    expect(request.method).toBe("POST");
    expect(request.credentials).toBe("same-origin");
    expect(Object.fromEntries(request.body.entries())).toEqual({ training_result_id: result.id, checkpoint, dataset_key: "снимки" });
    expect(props.closeModal).toHaveBeenCalledOnce();
    expect(props.reload).toHaveBeenCalledOnce();
  });

  it("использует лучший чекпойнт при отсутствии выбора в данных формы", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetcher);
    await submit(makeProps(), { image_folder_key: "орто" });
    expect(fetcher.mock.calls[0][1].body.get("checkpoint")).toBe("best");
    expect(fetcher.mock.calls[0][1].body.get("image_folder_key")).toBe("орто");
  });

  it("передаёт выбранный последний чекпойнт вместе с TXT, не меняя выбранную сеть", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetcher);
    const scenes = new File(["снимок"], "снимки.txt", { type: "text/plain" });
    await submit(makeProps(), { scenes_txt: scenes, checkpoint: "last" });
    const body = fetcher.mock.calls[0][1].body as FormData;
    expect(body.get("training_result_id")).toBe(result.id);
    expect(body.get("checkpoint")).toBe("last");
    expect((body.get("scenes_txt") as File).name).toBe("снимки.txt");
  });

  it("оставляет диалог и выбранный чекпойнт при ошибке отсутствующего файла", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Последний чекпойнт не сохранён." }), { status: 400 }));
    vi.stubGlobal("fetch", fetcher);
    const errors: unknown[] = [];
    const props = makeProps({ run: async <T,>(operation: () => Promise<T>) => {
      try { return await operation(); } catch (error) { errors.push(error); return undefined; }
    } });
    await submit(props, { dataset_key: "снимки", checkpoint: "last" });
    expect(errors).toHaveLength(0);
    expect(formState.updates).toContain("Последний чекпойнт не сохранён.");
    expect(props.closeModal).not.toHaveBeenCalled();
    expect(props.reload).not.toHaveBeenCalled();
    expect(fetcher).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls[0][1].body.get("checkpoint")).toBe("last");
  });

  it("передаёт потерю авторизации общему обработчику входа", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Войдите в систему." }), { status: 401 })));
    const errors: unknown[] = [];
    const props = makeProps({ run: async <T,>(operation: () => Promise<T>) => {
      try { return await operation(); } catch (error) { errors.push(error); return undefined; }
    } });
    await submit(props, { dataset_key: "снимки", checkpoint: "last" });
    expect(errors[0]).toBeInstanceOf(ApiError);
    expect(errors[0]).toMatchObject({ status: 401 });
    expect(props.closeModal).not.toHaveBeenCalled();
  });

  it("по-прежнему блокирует запуск без назначения шаблона инференса", async () => {
    const fetcher = vi.fn();
    vi.stubGlobal("fetch", fetcher);
    const props = makeProps({ inferenceAvailable: false });
    expect(renderToStaticMarkup(<PseudoMarkupForm {...props} />)).toContain('type="submit" disabled=""');
    await submit(props, { dataset_key: "снимки", checkpoint: "last" });
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("публикует компактную новость с постоянным адресом, сохраняя прежнее исправление", () => {
    const news = newsArticles.find(item => item.slug === "pseudo-markup-checkpoint-selection");
    expect(news?.date).toBe("2026-10-14");
    expect(news && "updates" in news && news.updates[0]).toContain("По умолчанию используются лучшие веса");
    expect(newsArticles.some(item => item.slug === "inference-compact-filter-direction")).toBe(true);
  });
});
