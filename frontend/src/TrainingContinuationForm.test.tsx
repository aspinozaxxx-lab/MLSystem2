import type { FormEvent } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { TrainingContinuationForm } from "./TrainingContinuationForm";
import type { TrainingResultInfo } from "./api/types";
import { newsArticles } from "./News";

const result: TrainingResultInfo = {
  comment: "",
  id: "учебная-сеть", model_name: "SegFormer B0", architecture: "smp_segformer_b0", input_channels: 4,
  created_at: "2026-10-15T09:00:00Z", source: "manual", status: "ok", pipeline_variant: "next_gen2",
  quality_metric: "pixel", validation_fold: 0, is_primary: false, can_continue_training: true, task: "binary",
};
const props = () => ({
  result, options: { additional_epochs: 10, additional_time_sec: 600, early_stopping_patience: 3,
    last_checkpoint_available: true, run_inference_after_training: false, secondary_priority: false },
  run: async <T,>(operation: () => Promise<T>) => operation(), closeModal: vi.fn(), reload: vi.fn(async () => {}),
});
const NativeFormData = FormData;
afterEach(() => { vi.unstubAllGlobals(); });

describe("состав датасета при дообучении", () => {
  it("оставляет прежний состав по умолчанию и предлагает актуальный", () => {
    const html = renderToStaticMarkup(<TrainingContinuationForm {...props()} />);
    expect(html).toContain("На каком составе датасета учить");
    expect(html).toContain('name="dataset_source" checked="" value="original"');
    expect(html).toContain('name="dataset_source" value="current"');
    expect(html).toContain("Последующие изменения датасета не учитываются");
  });

  it.each(["original", "current", undefined])("отправляет выбранный состав %s в API", async (source) => {
    const currentTarget = {} as HTMLFormElement;
    vi.stubGlobal("FormData", class extends NativeFormData {
      constructor(form?: HTMLFormElement) {
        super();
        if (form === currentTarget) {
          this.set("epochs", "10"); this.set("minutes", "10"); this.set("patience", "3");
          if (source) this.set("dataset_source", source);
        }
      }
    });
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({id:"задание"})));
    vi.stubGlobal("fetch", fetcher);
    const input = props();
    let onSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
    function CaptureForm() {
      const form = TrainingContinuationForm(input);
      onSubmit = form.props.onSubmit;
      return form;
    }
    renderToStaticMarkup(<CaptureForm />);
    await onSubmit!({preventDefault:vi.fn(), currentTarget} as unknown as FormEvent<HTMLFormElement>);
    const body = JSON.parse(fetcher.mock.calls[0][1].body);
    expect(body.dataset_source).toBe(source || "original");
    expect(body.checkpoint).toBe("best");
    expect(input.closeModal).toHaveBeenCalledOnce();
  });

  it("публикует инструкцию с датой, постоянным адресом и схемой", () => {
    const article = newsArticles.find(item => item.slug === "training-continuation-dataset-source");
    expect(article?.date).toBe("2026-10-15");
    if (!article || !("sections" in article)) throw new Error("Статья не найдена");
    expect(article.sections.length).toBeGreaterThanOrEqual(4);
    expect(article.image?.alt).toContain("Схема");
    expect(article.image?.src).toBe("/news/training-continuation-dataset-source.svg");
  });
});
