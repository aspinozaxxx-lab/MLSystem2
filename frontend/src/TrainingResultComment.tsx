import { PencilLine } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ApiError, apiJson } from "./api/client";
import type { TrainingResultCommentInfo } from "./api/types";

/** Неброская общая заметка к одному этапу, без модального окна и кнопки сохранения. */
export function TrainingResultComment({ resultId, comment = "", onSaved }: {
  resultId: string;
  comment?: string;
  onSaved?: (comment: string) => void;
}) {
  const [saved, setSaved] = useState(comment);
  const [draft, setDraft] = useState(comment);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const base = useRef(comment);
  const cancelBlur = useRef(false);
  const pending = useRef(false);
  const endpoint = `/results/training/${encodeURIComponent(resultId)}/comment`;

  useEffect(() => { setSaved(comment); }, [comment]);
  useEffect(() => {
    if (!editing) return;
    input.current?.focus();
  }, [editing]);
  useEffect(() => {
    const element = input.current;
    if (!element) return;
    element.style.height = "0px";
    element.style.height = `${element.scrollHeight}px`;
  }, [draft, editing]);

  const edit = () => {
    if (pending.current) return;
    base.current = saved;
    if (!error) setDraft(saved);
    setError("");
    setEditing(true);
  };
  const save = async () => {
    if (cancelBlur.current) { cancelBlur.current = false; return; }
    if (pending.current) return;
    const value = draft.trim();
    setEditing(false);
    if (value === base.current) { setError(""); return; }
    pending.current = true;
    setSaving(true);
    setError("");
    try {
      const response = await apiJson<TrainingResultCommentInfo>(endpoint, {
        method: "PUT", body: { comment: value, expected_comment: base.current },
      });
      setSaved(response.comment || "");
      setDraft(response.comment || "");
      onSaved?.(response.comment || "");
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 409) {
        try {
          const current = await apiJson<TrainingResultCommentInfo>(endpoint);
          setSaved(current.comment || "");
          onSaved?.(current.comment || "");
        } catch { /* Черновик остаётся доступным при повторной ошибке связи. */ }
      }
      setError(failure instanceof ApiError && failure.status === 409
        ? "Комментарий уже изменён. Ваш текст не сохранён: нажмите карандаш, чтобы повторить."
        : "Не удалось сохранить. Ваш текст остался здесь: нажмите карандаш и повторите.");
    } finally {
      pending.current = false;
      setSaving(false);
    }
  };

  return <small className="muted training-result-comment" aria-busy={saving}>
    {editing ? <textarea ref={input} className="training-comment-input" rows={1} maxLength={2000}
      aria-label="Комментарий к обучению" value={draft} onChange={event => setDraft(event.target.value)}
      onBlur={() => void save()} onKeyDown={event => {
        if (event.nativeEvent.isComposing) return;
        if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.blur(); }
        if (event.key === "Escape") {
          event.preventDefault(); cancelBlur.current = true;
          setDraft(saved); setError(""); setEditing(false); event.currentTarget.blur();
        }
      }} /> : <button className="training-comment-display" type="button" onClick={edit} disabled={saving}
        title="Редактировать комментарий" aria-label={saved || error ? "Редактировать комментарий к обучению" : "Добавить комментарий к обучению"}>
        {(error || saving ? draft : saved) || null}
        <PencilLine className="training-comment-pencil" size="1em" aria-hidden="true" />
      </button>}
    {saving ? <span className="training-comment-message" role="status">Сохранение…</span> : null}
    {error ? <span className="training-comment-message" role="alert">{error}</span> : null}
  </small>;
}
