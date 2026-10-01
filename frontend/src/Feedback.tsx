import { MessageSquarePlus, Send, X } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { apiJson } from "./api/client";
import type { FeedbackCreate, FeedbackInfo, FeedbackListResponse } from "./api/types";
import { formatDateTime } from "./utils/format";

export const feedbackStatusLabels = {
  waiting: "Ожидание",
  preparing: "Началась подготовка",
  implementing: "Реализация",
  implemented: "Реализовано",
} as const;

const kindLabels = { remark: "Замечание", improvement: "Улучшение", feature: "Новая функция" } as const;
const pageTitles: Record<string, string> = {
  start: "Запуск обучения", queue: "Очередь", templates: "Шаблоны", automation: "Автоматизация",
  classes: "Редактор классов", "dataset-editor": "Редактор датасетов", "pseudo-markup": "Просмотр псевдоразметки",
  results: "Результаты", jobs: "Задание", "model-export": "Экспорт модели", "scene-list-export": "Список сцен",
  "test-markups": "Тестовые разметки", news: "Новости", feedback: "Обращение",
};

export function feedbackPageContext(hash: string): { page_path: string; page_title: string } {
  const path = hash.startsWith("#/") ? hash.split("?")[0] : "#/";
  return { page_path: path, page_title: pageTitles[path.slice(2).split("/")[0]] || "Главная страница" };
}

export function FeedbackButton() {
  const [context, setContext] = useState<ReturnType<typeof feedbackPageContext> | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sent, setSent] = useState<FeedbackInfo | null>(null);
  const submission = useRef<string | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const firstField = useRef<HTMLTextAreaElement>(null);
  const close = useCallback(() => {
    if (busy) return;
    setContext(null);
    trigger.current?.focus();
  }, [busy]);
  useEffect(() => {
    if (!context) return;
    firstField.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
      if (event.key === "Tab") {
        const elements = [...document.querySelectorAll<HTMLElement>('.feedback-dialog button:not(:disabled), .feedback-dialog textarea:not(:disabled), .feedback-dialog a[href]')];
        const first = elements[0], last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => { document.removeEventListener("keydown", handleKey); document.body.style.overflow = previousOverflow; };
  }, [context, close]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!context || busy) return;
    setBusy(true); setError("");
    submission.current ||= crypto.randomUUID();
    try {
      const result = await apiJson<FeedbackInfo>("/feedback", { method: "POST", body: {
        submission_id: submission.current, kind: "improvement",
        title: message.trim().replace(/\s+/g, " ").slice(0, 160), message, ...context,
        credit_name: null,
      } satisfies FeedbackCreate });
      if (!result) throw new Error("Войдите в Гровику и повторите отправку");
      setSent(result); setMessage(""); submission.current = null;
      window.dispatchEvent(new Event("grovika-feedback-changed"));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось отправить обращение"); }
    finally { setBusy(false); }
  };

  return <>
    <button ref={trigger} type="button" title="Предложить улучшение" aria-label="Предложить улучшение" onClick={() => {
      setContext(feedbackPageContext(window.location.hash)); setSent(null); setError("");
    }}><MessageSquarePlus size={16} /><span className="nav-label">Предложить улучшение</span></button>
    {context && createPortal(<div className="feedback-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
      <section className="panel feedback-dialog" role="dialog" aria-modal="true" aria-labelledby="feedback-dialog-title">
        <div className="feedback-heading"><h2 id="feedback-dialog-title">Предложить улучшение</h2><button type="button" aria-label="Закрыть обращение" disabled={busy} onClick={close}><X size={18} /></button></div>
        {sent ? <div className="form-stack" role="status"><p>Обращение №{sent.id} отправлено. Его подготовка начнётся после ближайшей проверки.</p><a className="primary" href={`#/feedback/${sent.id}`} onClick={close}>Следить за обращением</a><p className="muted">Все предложения и их этапы видны на главной странице перед новостями.</p></div> : <form className="form-stack" onSubmit={submit}>
          <p className="muted">Страница: {context.page_title}. Автор и адрес страницы сохранятся автоматически.</p>
          <label className="field"><span>Опишите своё предложение</span><textarea ref={firstField} value={message} onChange={(event) => setMessage(event.target.value)} minLength={5} maxLength={6000} rows={5} disabled={busy} required /></label>
          <p className="muted">Текст будет виден всем вошедшим пользователям. Не включайте пароли и личные данные.</p>
          {error && <p className="error-box" role="alert">{error}</p>}
          <button className="primary" type="submit" disabled={busy}><Send size={16} />{busy ? "Отправка…" : "Отправить"}</button>
        </form>}
      </section>
    </div>, document.body)}
  </>;
}

export function FeedbackCard({ item }: { item: FeedbackInfo }) {
  return <article className="feedback-card" id={`proposal-${item.id}`}>
    <div className="feedback-card-header"><a className="feedback-title" href={`#/feedback/${item.id}`}>№{item.id} · {item.title}</a><span className={`feedback-status feedback-status-${item.status}`}>{feedbackStatusLabels[item.status]}</span></div>
    <p className="muted feedback-meta">{kindLabels[item.kind]} · {item.author} · {formatDateTime(item.created_at)} · <a href={item.page_path}>{item.page_title}</a></p>
    {item.progress && <p className="feedback-progress">{item.progress}</p>}
    {item.status === "preparing" && item.preparation && !item.progress.includes("Ожидается подтверждение") && <p className="muted">Решение подготовлено. Ожидается подтверждение владельца.</p>}
    <details><summary>Подробности предложения</summary><p className="feedback-text">{item.message}</p>{item.preparation && <><h3>Подготовленное решение</h3><p className="feedback-text">{item.preparation}</p></>}{item.approved_at && <p className="muted">Реализация подтверждена {formatDateTime(item.approved_at)}.</p>}</details>
    {item.news_slug && <a className="feedback-result" href={`#/news/${item.news_slug}`}>Что изменилось — читать новость</a>}
  </article>;
}

export function FeedbackSection({ feedbackId }: { feedbackId?: string }) {
  const [items, setItems] = useState<FeedbackInfo[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadingMore, setLoadingMore] = useState(false);
  const loadedPages = useRef(1);
  const load = useCallback(async () => {
    try {
      if (feedbackId) {
        const item = await apiJson<FeedbackInfo>(`/feedback/${encodeURIComponent(feedbackId)}`);
        if (item) setItems([item]);
      } else {
        const refreshed: FeedbackInfo[] = [];
        let cursor = "", morePages = false;
        for (let page = 0; page < loadedPages.current; page++) {
          const result = await apiJson<FeedbackListResponse>(`/feedback?limit=20${cursor}`);
          if (!result) return;
          refreshed.push(...result.items);
          morePages = result.has_more;
          if (!morePages || !result.items.length) break;
          cursor = `&before_id=${result.items[result.items.length - 1].id}`;
        }
        setItems(refreshed); setHasMore(morePages);
      }
      setError("");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось загрузить предложения"); }
    finally { setLoading(false); }
  }, [feedbackId]);
  useEffect(() => {
    setItems([]); setLoading(true); loadedPages.current = 1;
    void load();
    const timer = window.setInterval(() => { if (document.visibilityState === "visible") void load(); }, 30_000);
    window.addEventListener("grovika-feedback-changed", load);
    return () => { window.clearInterval(timer); window.removeEventListener("grovika-feedback-changed", load); };
  }, [load]);
  const more = async () => {
    setLoadingMore(true);
    try {
      const result = await apiJson<FeedbackListResponse>(`/feedback?limit=20&before_id=${items[items.length - 1].id}`);
      if (result) { setItems((previous) => [...previous, ...result.items.filter((item) => !previous.some((old) => old.id === item.id))]); setHasMore(result.has_more); loadedPages.current += 1; }
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Не удалось загрузить предложения"); }
    finally { setLoadingMore(false); }
  };
  return <section className="panel feedback-section" aria-labelledby="feedback-heading">
    <div className="feedback-heading"><h2 id="feedback-heading">{feedbackId ? `Обращение №${feedbackId}` : "Предложения пользователей"}</h2>{feedbackId && <a href="#/">Все предложения</a>}</div>
    <p className="muted">Ожидание → Началась подготовка → Реализация после подтверждения → Реализовано</p>
    {error && <div className="error-box" role="alert">{error} <button type="button" onClick={() => void load()}>Повторить</button></div>}
    {loading ? <p role="status">Загрузка предложений…</p> : !items.length && !error ? <p className="muted">Пока предложений нет. Кнопка «Предложить улучшение» в меню доступна на каждой странице.</p> : items.map((item) => <FeedbackCard key={item.id} item={item} />)}
    {!feedbackId && hasMore && items.length > 0 && <button type="button" disabled={loadingMore} onClick={() => void more()}>{loadingMore ? "Загрузка…" : "Показать ещё"}</button>}
  </section>;
}
