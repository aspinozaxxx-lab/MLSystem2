import { FileText } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

type ClassChoice = { key: string; name: string; count?: number; draftCount?: number };

export function useRememberedClass(classes: readonly { key: string }[] | null, storageKey: string) {
  const [savedKey, setSavedKey] = useState(() => {
    try { return localStorage.getItem(storageKey) || ""; } catch { return ""; }
  });
  const latestKey = useRef(savedKey);
  const classKey = classes === null ? savedKey
    : classes.some((item) => item.key === savedKey) ? savedKey : classes[0]?.key || "";
  const setClassKey = useCallback((key: string) => {
    latestKey.current = key;
    setSavedKey(key);
    try { localStorage.setItem(storageKey, key); } catch { /* Выбор доступен без хранилища браузера. */ }
  }, [storageKey]);
  useEffect(() => {
    // Поздний ответ каталога не отменяет уже сделанный пользователем выбор.
    if (classes !== null && classKey !== savedKey && latestKey.current === savedKey) setClassKey(classKey);
  }, [classes, classKey, savedKey, setClassKey]);
  return { classKey, setClassKey };
}

export function ClassNavigation({ classes, classKey, onSelect, label }: {
  classes: readonly ClassChoice[] | null;
  classKey: string;
  onSelect: (key: string) => void;
  label: string;
}) {
  return <>
    <nav className="class-browser-nav" aria-label={label}>
      <span className="field-label">Классы</span>
      {classes?.map((item) => <button key={item.key} type="button" title={item.name}
        className={classKey === item.key ? "selected" : ""}
        aria-current={classKey === item.key ? "true" : undefined} onClick={() => onSelect(item.key)}>
        <span>{item.name}</span>
        <small>
          {item.draftCount ? <span className="class-browser-draft-count" title={`Черновиков: ${item.draftCount}`}><FileText size={11} />{item.draftCount}</span> : null}
          {item.count}
        </small>
      </button>)}
      {classes === null ? <span className="muted" role="status">Загрузка классов…</span> : null}
    </nav>
    <label className="field class-browser-mobile"><span>Класс</span>
      <select aria-label={label} value={classKey} onChange={(event) => onSelect(event.target.value)} disabled={!classes?.length}>
        {!classes?.length ? <option value="">{classes === null ? "Загрузка…" : "Нет классов"}</option> : null}
        {classes?.map((item) => <option key={item.key} value={item.key}>
          {item.name}{item.count !== undefined ? ` · ${item.count}` : ""}{item.draftCount ? ` · черновиков: ${item.draftCount}` : ""}
        </option>)}
      </select>
    </label>
  </>;
}
