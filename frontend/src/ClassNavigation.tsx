import { FileText } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";

type ClassChoice = { key: string; name: string; count?: number; draftCount?: number };

const storageKey = "grovika:selected-class";
let rememberedClassKey: string | undefined;
const listeners = new Set<() => void>();

function readRememberedClass(): string {
  if (rememberedClassKey === undefined) {
    try {
      rememberedClassKey = localStorage.getItem(storageKey)
        // Однократно подхватываем прежний выбор при обновлении приложения.
        || localStorage.getItem("grovika:results:class")
        || localStorage.getItem("grovika:test-markups:class")
        || localStorage.getItem("grovika:class-editor:class") || "";
    } catch { rememberedClassKey = ""; }
  }
  return rememberedClassKey;
}

function rememberClass(key: string) {
  rememberedClassKey = key;
  try { localStorage.setItem(storageKey, key); } catch { /* Выбор сохраняется в текущей вкладке. */ }
  listeners.forEach((notify) => notify());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function useRememberedClass(classes: readonly { key: string }[] | null) {
  const savedKey = useSyncExternalStore(subscribe, readRememberedClass, readRememberedClass);
  const classKey = classes === null ? savedKey
    : classes.some((item) => item.key === savedKey) ? savedKey : classes[0]?.key || "";
  useEffect(() => {
    // Локальный запасной класс не отменяет общий выбор в ограниченном каталоге.
    // Поздний ответ каталога также не отменяет уже сделанный выбор.
    if (!savedKey && classKey && !readRememberedClass()) rememberClass(classKey);
  }, [classKey, savedKey]);
  return { classKey, setClassKey: rememberClass };
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
