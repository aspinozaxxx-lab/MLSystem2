import { Download } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { TestSampleDownloadRequest } from "./api/types";

export type TestSampleAnnotationLayout = TestSampleDownloadRequest["annotation_layout"];

export function TestSampleDownloadOptionsForm({ onCancel, onSubmit }: {
  onCancel: () => void;
  onSubmit: (layout: TestSampleAnnotationLayout, includeImages: boolean) => Promise<boolean>;
}) {
  const [layout, setLayout] = useState<TestSampleAnnotationLayout>("single");
  const [includeImages, setIncludeImages] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      if (await onSubmit(layout, includeImages)) onCancel();
    } finally {
      setSubmitting(false);
    }
  };
  return (
    <form className="form-stack" onSubmit={submit}>
      <fieldset className="download-mode-fieldset" disabled={submitting}>
        <legend>Формат разметки</legend>
        <label className="download-mode-choice">
          <input type="radio" name="download-mode" value="per_image"
            checked={layout === "per_image"} onChange={() => setLayout("per_image")} />
          <span><strong>GeoJSON на каждый снимок</strong><small>Отдельный файл исходных контуров к каждому снимку</small></span>
        </label>
        <label className="download-mode-choice">
          <input type="radio" name="download-mode" value="single"
            checked={layout === "single"} onChange={() => setLayout("single")} />
          <span><strong>Один GeoJSON</strong><small>Общая разметка: наложения и общие рёбра объектов одного типа склеены</small></span>
        </label>
      </fieldset>
      <fieldset className="download-mode-fieldset" disabled={submitting}>
        <legend>Снимки</legend>
        <label className="download-mode-choice">
          <input type="checkbox" checked={includeImages} onChange={event => setIncludeImages(event.target.checked)} />
          <span><strong>Добавить снимки TIFF</strong><small>По умолчанию архив содержит только GeoJSON. Снимки увеличивают объём скачивания.</small></span>
        </label>
      </fieldset>
      <div className="button-row download-dialog-actions">
        <button className="secondary" type="button" disabled={submitting} onClick={onCancel}>Отмена</button>
        <button className="primary" type="submit" disabled={submitting}>
          <Download size={16} />{submitting ? "Формирование…" : "Скачать ZIP"}
        </button>
      </div>
    </form>
  );
}
