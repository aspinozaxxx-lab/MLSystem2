import { useEffect, useState } from "react";
import { History, X } from "lucide-react";
import { apiJson } from "./api/client";
import type { AnnotationFeatureHistory, AnnotationFeatureHistoryEntry } from "./api/types";
import { formatDateTime } from "./utils/format";
import type { JsonObject } from "./utils/datasetEditor";

export type PolygonHistoryPreview = { feature: JsonObject; crs: JsonObject | null | undefined; before: boolean }[];
const actionNames = { baseline: "Исходный контур", created: "Добавлен в версию", updated: "Изменён", deleted: "Удалён" };

export function polygonHistoryPreview(entries: AnnotationFeatureHistoryEntry[], index: number): PolygonHistoryPreview {
  const current = entries[index], previous = entries[index - 1];
  return [previous?.feature ? { feature: previous.feature, crs: previous.crs, before: true } : null,
    current?.feature ? { feature: current.feature, crs: current.crs, before: false } : null]
    .filter((item): item is PolygonHistoryPreview[number] => item !== null);
}

export function DatasetPolygonHistory({ path, onPreview }: { path: string | null; onPreview: (preview: PolygonHistoryPreview) => void }) {
  const [history, setHistory] = useState<AnnotationFeatureHistory | null>(null);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => {
    let active = true;
    setHistory(null); setError(""); setExpanded(false); setSelected(null); onPreview([]);
    if (path) void apiJson<AnnotationFeatureHistory>(path).then((result) => { if (active) setHistory(result); })
      .catch((reason) => { if (active) setError(reason instanceof Error ? reason.message : "Не удалось прочитать историю."); });
    return () => { active = false; onPreview([]); };
  }, [path, onPreview]);
  const created = history?.entries.find((entry) => entry.action === "created");
  const changed = history?.entries.filter((entry) => entry.action !== "baseline").at(-1);
  return <section className="polygon-history" aria-label="История выбранного полигона">
    <header><strong><History size={14} /> История полигона</strong>
      {history?.entries.length ? <button type="button" className="secondary" aria-expanded={expanded}
        onClick={() => { setExpanded(!expanded); setSelected(null); onPreview([]); }}>{expanded ? "Свернуть" : "Изменения и контуры"}</button> : null}</header>
    {!path ? <p>История появится после публикации новой версии или снимка.</p> : error ? <p role="alert">{error}</p> : !history ? <p role="status">Загрузка истории…</p> : <>
      {!history.entries.length ? <p>Полигон ещё не опубликован. Личные черновики в историю не входят.</p> : <>
        <p>{created ? `Добавлен в эту версию: ${created.username}, ${formatDateTime(created.published_at!)}.` : "Автор и дата создания прежнего полигона неизвестны: разметка импортирована до начала учёта истории."}</p>
        {changed ? <p>Последняя публикация: {changed.username}, {formatDateTime(changed.published_at!)}. Черновые изменения не учитываются.</p> : null}
      </>}
      {expanded ? <>
        <div className="polygon-history-events">{history.entries.map((entry, index) => <button type="button" key={entry.id}
          className={selected === index ? "primary" : "secondary"} aria-pressed={selected === index}
          onClick={() => { setSelected(index); onPreview(polygonHistoryPreview(history.entries, index)); }}>
          <strong>{actionNames[entry.action]}</strong>
          <small>{entry.published_at ? `${entry.username} · ${formatDateTime(entry.published_at)}` : "Автор и дата неизвестны"}</small>
          {entry.commit && entry.action !== "baseline" ? <small>Публикация {entry.commit.slice(0, 8)}</small> : null}
        </button>)}</div>
        {selected !== null ? <p>На карте: голубой пунктир — до изменения, розовый — после. Это просмотр, текущая разметка не меняется. <button type="button" className="secondary icon-button" aria-label="Убрать сравнение контуров" onClick={() => { setSelected(null); onPreview([]); }}><X size={13} /></button></p> : null}
      </> : null}
    </>}
  </section>;
}
