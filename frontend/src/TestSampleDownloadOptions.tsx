import { Download } from "lucide-react";
import { type FormEvent, useState } from "react";
import type { TestSampleDownloadRequest } from "./api/types";

export type TestSampleAnnotationLayout = TestSampleDownloadRequest["annotation_layout"];

export function TestSampleDownloadOptionsForm({ onCancel, onSubmit }: {
  onCancel: () => void;
  onSubmit: (layout: TestSampleAnnotationLayout) => Promise<boolean>;
}) {
  const [layout, setLayout] = useState<TestSampleAnnotationLayout>("single");
  const [submitting, setSubmitting] = useState(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      if (await onSubmit(layout)) onCancel();
    } finally {
      setSubmitting(false);
    }
  };
  return (
    <form className="form-stack" onSubmit={submit}>
      <fieldset className="download-mode-fieldset" disabled={submitting}>
        <legend>Состав архива</legend>
        <label className="download-mode-choice">
          <input type="radio" name="download-mode" value="per_image"
            checked={layout === "per_image"} onChange={() => setLayout("per_image")} />
          <span><strong>GeoJSON на каждый снимок + TIFF</strong><small>Отдельный файл исходных контуров к каждому снимку</small></span>
        </label>
        <label className="download-mode-choice">
          <input type="radio" name="download-mode" value="single"
            checked={layout === "single"} onChange={() => setLayout("single")} />
          <span><strong>Один GeoJSON + TIFF</strong><small>Общая разметка: наложения и общие рёбра объектов одного типа склеены</small></span>
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
