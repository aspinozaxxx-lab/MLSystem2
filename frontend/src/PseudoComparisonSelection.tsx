import { ListPlus, Trash2, X } from "lucide-react";
import { createContext, useContext, useState, type ReactNode } from "react";

type SelectionItem = { id: string; label: string };
type Selection = { items: SelectionItem[]; toggle: (item: SelectionItem) => void; clear: () => void };
const SelectionContext = createContext<Selection>({ items: [], toggle: () => {}, clear: () => {} });

export function PseudoComparisonProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<SelectionItem[]>([]);
  return <SelectionContext value={{ items, clear: () => setItems([]), toggle: (item) => setItems((previous) =>
    previous.some((value) => value.id === item.id) ? previous.filter((value) => value.id !== item.id)
      : previous.length < 12 ? [...previous, item] : previous) }}>{children}</SelectionContext>;
}

export function PseudoCompareButton({ id, label }: SelectionItem) {
  const { items, toggle } = useContext(SelectionContext);
  const selected = items.some((item) => item.id === id);
  return <button className={`${selected ? "primary" : "secondary"} icon-button`} type="button"
    aria-label={`${selected ? "Убрать из сравнения" : "Добавить в сравнение"}: ${label}`} aria-pressed={selected}
    title={selected ? "Убрать из сравнения" : "Добавить в список сравнения псевдоразметок"}
    disabled={!selected && items.length >= 12} onClick={() => toggle({ id, label })}><ListPlus size={15} /></button>;
}

export function PseudoComparisonTray() {
  const { items, toggle, clear } = useContext(SelectionContext);
  if (!items.length) return null;
  return <section className="panel pseudo-comparison-tray" aria-label="Список сравнения псевдоразметок">
    <div className="pseudo-comparison-tray-heading"><strong>Сравнение псевдоразметок · {items.length}/12</strong>
      <div className="button-row">{items.length >= 2 ? <a className="primary compact-action" href={`#/pseudo-markup/compare/${items.map((item) => item.id).join(",")}`}>Сравнить</a>
        : <button className="primary compact-action" type="button" disabled>Выберите ещё одну</button>}
        <button className="secondary icon-button" type="button" aria-label="Очистить список сравнения" title="Очистить список сравнения" onClick={clear}><Trash2 size={16} /></button></div>
    </div>
    <div className="pseudo-comparison-chips">{items.map((item, index) => <span key={item.id}><span title={item.label}>{index + 1}. {item.label}</span>
      <button type="button" className="secondary icon-button" aria-label={`Убрать из сравнения: ${item.label}`} onClick={() => toggle(item)}><X size={14} /></button></span>)}</div>
    <small className="muted">Добавляйте готовые разметки из любых классов и датасетов. Список сохраняется при переходах между результатами до выхода из Гровики.</small>
  </section>;
}
