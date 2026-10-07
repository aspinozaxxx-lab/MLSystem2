import { AlertTriangle, ArrowRight, Layers3, LoaderCircle, PencilLine, Trash2 } from "lucide-react";
import { type FormEvent, type ReactNode, useEffect, useLayoutEffect, useRef, useState } from "react";
import { apiJson } from "./api/client";
import type { BootstrapInfo, InferenceTemplate, JsonRecord } from "./api/types";
import { ConfigEditor } from "./ConfigEditor";

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;
type Modal = { title: string; body: ReactNode; footer?: ReactNode; wide?: boolean };

function countLabel(count: number, forms: [string, string, string]) {
  const last = count % 10;
  return `${count} ${forms[count % 100 >= 11 && count % 100 <= 14 ? 2 : last === 1 ? 0 : last >= 2 && last <= 4 ? 1 : 2]}`;
}

export function InferenceTemplates({ bootstrap, run, reload, showModal, closeModal, renderHeader }: {
  bootstrap: BootstrapInfo; run: Runner; reload: () => Promise<void>;
  showModal: (modal: Modal) => void; closeModal: () => void;
  renderHeader: (create: () => void) => ReactNode;
}) {
  const templates = bootstrap.inference_templates;
  const classes = bootstrap.classes.filter((item) => item.key !== "custom");
  const [menuClass, setMenuClass] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState({ left: 0, top: 0 });
  const menuElement = useRef<HTMLDivElement>(null);
  const [moving, setMoving] = useState<string | null>(null);
  useLayoutEffect(() => {
    const element = menuElement.current;
    if (!element) return;
    const height = Math.min(element.scrollHeight, window.innerHeight - 24);
    setMenuPosition((position) => ({ ...position, top: Math.max(12, Math.min(position.top, window.innerHeight - height - 12)) }));
  }, [menuClass]);
  useEffect(() => {
    if (!menuClass) return;
    const close = (event: PointerEvent) => {
      if (!(event.target as Element).closest(".inference-class-item")) setMenuClass(null);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setMenuClass(null); };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", escape);
    };
  }, [menuClass]);
  const unassigned = classes.filter((item) => !templates.some((template) => template.class_keys.includes(item.key)));
  const move = async (classKey: string, templateId: string | null) => {
    if (moving) return;
    setMoving(classKey);
    setMenuClass(null);
    try {
      const updated = await run(() => apiJson(`/dataset-classes/${encodeURIComponent(classKey)}/inference-template`, {
        method: "PUT", body: { template_id: templateId },
      }));
      if (updated) await reload();
    } finally { setMoving(null); }
  };
  const settings = (template: InferenceTemplate) => showModal({
    title: "Настроить шаблон инференса", wide: true, footer: null,
    body: <InferenceTemplateForm template={template} run={run} reload={reload} close={closeModal} />,
  });
  const create = () => showModal({
    title: "Новый шаблон инференса", footer: null,
    body: <InferenceTemplateForm run={run} reload={reload} close={closeModal} onCreated={settings} />,
  });
  const remove = (template: InferenceTemplate) => {
    const attached = classes.filter((item) => template.class_keys.includes(item.key));
    showModal({
      title: `Удалить «${template.display_name}»`,
      body: <div className="form-stack"><p>Настройки шаблона будут удалены из списка.</p>{attached.length ?
        <div className="inference-template-warning">Классы {attached.map((item) => `«${item.name}»`).join(", ")} останутся без шаблона. Создание псевдоразметки станет недоступно до нового назначения.</div> : null}</div>,
      footer: <><button type="button" className="secondary" onClick={closeModal}>Отмена</button>
        <button type="button" className="danger" onClick={async () => {
          const deleted = await run(() => apiJson(`/inference-templates/by-id/${template.id}`, { method: "DELETE" }));
          if (deleted) { closeModal(); await reload(); }
        }}><Trash2 size={16} />{attached.length ? "Удалить и снять привязки" : "Удалить"}</button></>,
    });
  };
  const chips = (keys: string[], current: string | null) => <div className="inference-class-chips">
    {classes.filter((item) => keys.includes(item.key)).sort((a, b) => a.name.localeCompare(b.name, "ru")).map((item) =>
      <div className="inference-class-item" key={item.key}>
        <button type="button" className="badge source-badge manual inference-class-chip" aria-label={`Перенести класс «${item.name}»`}
          aria-haspopup="menu" aria-expanded={menuClass === item.key} disabled={Boolean(moving)}
          onClick={(event) => {
            const anchor = event.currentTarget.getBoundingClientRect();
            setMenuPosition({ left: Math.max(12, Math.min(anchor.left, window.innerWidth - 282)), top: anchor.bottom + 7 });
            setMenuClass(menuClass === item.key ? null : item.key);
          }}>
          <span>{item.name}</span>{moving === item.key ? <LoaderCircle size={12} className="status-spinner" /> : null}
        </button>
        {menuClass === item.key ? <div ref={menuElement} className="inference-chip-menu" role="menu" aria-label={`Шаблон для класса «${item.name}»`} style={{ ...menuPosition, maxHeight: `calc(100dvh - ${menuPosition.top + 12}px)` }}>
          <span className="inference-menu-caption">Перенести в шаблон</span>
          {templates.filter((target) => target.id !== current && target.is_active).map((target) =>
            <button type="button" role="menuitem" key={target.id} onClick={() => void move(item.key, target.id)}>
              <ArrowRight size={14} /><span>{target.display_name}</span>
            </button>)}
          {!templates.some((target) => target.id !== current && target.is_active) ? <small>Создайте ещё один шаблон для переноса.</small> : null}
          {current ? <button type="button" role="menuitem" className="inference-menu-detach" onClick={() => void move(item.key, null)}>Снять привязку</button> : null}
        </div> : null}
      </div>)}
  </div>;
  return <>
    {renderHeader(create)}
    <div className="inference-template-grid">
      {templates.map((template) => <section className={`panel inference-template-card${template.class_keys.length > 6 ? " has-many-classes" : ""}`} key={template.id}>
        <header><span className="inference-template-icon"><Layers3 size={22} /></span><div><h2>{template.display_name}</h2>
          <span className="muted">{countLabel(template.class_keys.length, ["класс", "класса", "классов"])}{template.is_active ? "" : " · недоступен"}</span></div>
          <div className="inference-card-actions"><button type="button" className="secondary icon-button" onClick={() => settings(template)} title="Настроить" aria-label={`Настроить «${template.display_name}»`}><PencilLine size={16} /></button>
            <button type="button" className="ghost icon-button" onClick={() => remove(template)} title="Удалить" aria-label={`Удалить «${template.display_name}»`}><Trash2 size={16} /></button></div>
        </header>
        {template.description ? <p className="inference-template-description">{template.description}</p> : null}
        {template.class_keys.length ? chips(template.class_keys, template.id) : <div className="inference-empty-classes">Классы пока не назначены. Перенесите их сюда через меню плашки.</div>}
      </section>)}
      {unassigned.length ? <section className={`panel inference-template-card is-unassigned${unassigned.length > 6 ? " has-many-classes" : ""}`}>
        <header><span className="inference-template-icon"><AlertTriangle size={22} /></span><div><h2>Без шаблона</h2><span className="muted">{countLabel(unassigned.length, ["класс", "класса", "классов"])}</span></div></header>
        <p className="inference-template-description">Назначьте шаблон нажатием на плашку. Псевдоразметка для этих классов недоступна.</p>
        {chips(unassigned.map((item) => item.key), null)}
      </section> : null}
    </div>
  </>;
}

function InferenceTemplateForm({ template, run, reload, close, onCreated }: {
  template?: InferenceTemplate; run: Runner; reload: () => Promise<void>; close: () => void;
  onCreated?: (template: InferenceTemplate) => void;
}) {
  const [name, setName] = useState(template?.display_name || "");
  const [description, setDescription] = useState(template?.description || "");
  const [config, setConfig] = useState<JsonRecord>({ ...(template?.default_config || {}) });
  const [busy, setBusy] = useState(false);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (busy || !name.trim()) return;
    setBusy(true);
    try {
      const updated = await run(() => apiJson<InferenceTemplate>(template ? `/inference-templates/by-id/${template.id}` : "/inference-templates", {
        method: template ? "PUT" : "POST",
        body: { display_name: name.trim(), description: description.trim() || null, ...(template ? { default_config: config } : {}) },
      }));
      if (updated) { await reload(); if (onCreated) onCreated(updated); else close(); }
    } finally { setBusy(false); }
  };
  const reset = async () => {
    if (!template || busy) return;
    setBusy(true);
    try {
      const updated = await run(() => apiJson<InferenceTemplate>(`/inference-templates/by-id/${template.id}`, { method: "PUT", body: { reset_to_baseline: true } }));
      if (updated) { setConfig({ ...updated.default_config }); await reload(); }
    } finally { setBusy(false); }
  };
  return <form className="form-stack inference-template-form" onSubmit={save}>
    <label className="field"><span>Название</span><input value={name} onChange={(event) => setName(event.target.value)} required maxLength={240} disabled={busy} placeholder="Например, подробные контуры" /></label>
    <label className="field"><span>Описание <small>необязательно</small></span><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={2000} rows={2} disabled={busy} placeholder="Для каких объектов и задач подходят параметры" /></label>
    {template ? <><p className="info-box">Изменения применятся к будущим операциям всех привязанных классов. Готовые псевдоразметки и запущенные задания сохраняются.</p>
      <ConfigEditor schema={template.config_schema} value={config} onChange={setConfig} readonly={busy} presentation={{ "postprocess.filter_compact_objects.mode": { options: { remove_compact: "Убирать компактные", keep_compact: "Оставлять компактные" } } }} /></> : <p className="muted">После создания настройте параметры и назначьте классы через меню их плашек.</p>}
    <div className="button-row"><button type="submit" className="primary" disabled={busy || !name.trim()}>{busy ? <LoaderCircle size={16} className="status-spinner" /> : null}{busy ? "Сохранение…" : template ? "Сохранить" : "Создать и настроить"}</button>
      {template ? <button type="button" className="secondary" disabled={busy} onClick={() => void reset()}>Сбросить параметры</button> : null}
      <button type="button" className="secondary" disabled={busy} onClick={close}>Отмена</button></div>
  </form>;
}
