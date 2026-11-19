import { Download, LoaderCircle, Paperclip, Save, Trash2 } from "lucide-react";
import { useEffect, useRef, useState, type ChangeEvent } from "react";

import { apiForm, apiJson } from "./api/client";
import type { PersonalNoteAttachment, PersonalNoteInfo } from "./api/types";
import { formatFileSize } from "./utils/format";

const ENDPOINT = "/personal-note";
type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;

export function PersonalNotePage({ run, registerRouteGuard }: {
  run: Runner;
  registerRouteGuard: (guard: (() => boolean) | null) => void;
}) {
  const [note, setNote] = useState<PersonalNoteInfo | null>(null);
  const [draft, setDraft] = useState("");
  const [loadFailed, setLoadFailed] = useState(false);
  const [busy, setBusy] = useState("");
  const [uploadProgress, setUploadProgress] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const pending = useRef(false);
  const dirty = note !== null && draft !== note.content;

  useEffect(() => {
    let cancelled = false;
    void run(() => apiJson<PersonalNoteInfo>(ENDPOINT)).then(payload => {
      if (cancelled) return;
      if (!payload) { setLoadFailed(true); return; }
      setNote(payload);
      setDraft(payload.content);
    });
    return () => { cancelled = true; };
  }, [run]);

  useEffect(() => {
    registerRouteGuard(() => {
      if (pending.current) return false;
      return !dirty || window.confirm("В заметке есть несохранённые изменения. Покинуть страницу?");
    });
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirty && !pending.current) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => {
      registerRouteGuard(null);
      window.removeEventListener("beforeunload", beforeUnload);
    };
  }, [dirty, registerRouteGuard]);

  const save = async () => {
    if (!note || !dirty || pending.current) return;
    pending.current = true;
    setBusy("save");
    try {
      const saved = await run(() => apiJson<PersonalNoteInfo>(ENDPOINT, {
        method: "PUT", body: { content: draft, expected_revision: note.revision },
      }));
      if (saved) setNote(saved);
    } finally { pending.current = false; setBusy(""); }
  };

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    if (!files.length || pending.current) return;
    pending.current = true;
    setBusy("upload");
    try {
      for (const [index, file] of files.entries()) {
        const label = `Файл ${index + 1} из ${files.length}`;
        setUploadProgress(`${label}: загрузка…`);
        const form = new FormData();
        form.append("file", file);
        const saved = await run(() => apiForm<PersonalNoteAttachment>(`${ENDPOINT}/attachments`, form, {
          onUploadProgress: progress => setUploadProgress(progress.complete
            ? `${label}: сохранение…`
            : `${label}: ${progress.total ? Math.round(100 * progress.loaded / progress.total) : 0}%`),
        }));
        if (!saved) break;
        setNote(current => current ? { ...current, attachments: [...current.attachments, saved] } : current);
      }
    } finally { pending.current = false; setBusy(""); setUploadProgress(""); }
  };

  const remove = async (file: PersonalNoteAttachment) => {
    if (pending.current || !window.confirm(`Удалить файл «${file.original_name}» из заметки?`)) return;
    pending.current = true;
    setBusy(file.id);
    try {
      const removed = await run(async () => {
        await apiJson(`${ENDPOINT}/attachments/${encodeURIComponent(file.id)}`, { method: "DELETE" });
        return true;
      });
      if (removed) setNote(current => current ? {
        ...current, attachments: current.attachments.filter(item => item.id !== file.id),
      } : current);
    } finally { pending.current = false; setBusy(""); }
  };

  if (!note) return <p role="status">{loadFailed ? "Не удалось загрузить заметку. Обновите страницу, чтобы повторить." : "Загрузка заметки…"}</p>;
  return <section className="personal-note ym-hide-content">
    <h1>Заметки</h1>
    <PersonalNoteEditor
      content={draft} dirty={dirty} busy={busy} attachments={note.attachments} uploadProgress={uploadProgress}
      onChange={setDraft} onSave={() => void save()} onAttach={() => fileInput.current?.click()}
      onRemove={file => void remove(file)}
    />
    <input ref={fileInput} type="file" multiple hidden aria-label="Выбрать файлы для заметки" onChange={event => void upload(event)} />
  </section>;
}

export function PersonalNoteEditor({ content, dirty, busy, attachments, uploadProgress, onChange, onSave, onAttach, onRemove }: {
  content: string;
  dirty: boolean;
  busy: string;
  attachments: PersonalNoteAttachment[];
  uploadProgress: string;
  onChange: (content: string) => void;
  onSave: () => void;
  onAttach: () => void;
  onRemove: (file: PersonalNoteAttachment) => void;
}) {
  return <>
    <div className="personal-note-toolbar">
      <button type="button" className="primary" disabled={!dirty || Boolean(busy)} onClick={onSave}>
        {busy === "save" ? <LoaderCircle size={16} className="status-spinner" /> : <Save size={16} />}Сохранить
      </button>
      <span className={dirty ? "personal-note-status changed" : "personal-note-status muted"} role="status" aria-live="polite">
        {busy === "save" ? "Сохранение…" : dirty ? "Есть несохранённые изменения" : "Все изменения сохранены"}
      </span>
    </div>
    <textarea className="personal-note-text" aria-label="Текст заметки" value={content}
      onChange={event => onChange(event.target.value)} spellCheck={false} />
    <div className="personal-note-toolbar">
      <button type="button" className="secondary" disabled={Boolean(busy)} onClick={onAttach}><Paperclip size={16} />Прикрепить файлы</button>
      <small className="muted" role="status">{uploadProgress || "Вложения сохраняются сразу"}</small>
    </div>
    {attachments.length ? <ul className="personal-note-files" aria-label="Файлы заметки">
      {attachments.map(file => <li key={file.id}>
        <a href={file.download_url} download><Download size={16} /><span>{file.original_name}</span></a>
        <small className="muted">{formatFileSize(file.size_bytes)}</small>
        <button type="button" className="icon-button danger" disabled={Boolean(busy)}
          aria-label={`Удалить файл ${file.original_name}`} title="Удалить файл" onClick={() => onRemove(file)}>
          {busy === file.id ? <LoaderCircle size={16} className="status-spinner" /> : <Trash2 size={16} />}
        </button>
      </li>)}
    </ul> : null}
  </>;
}
