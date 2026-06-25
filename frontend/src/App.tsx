import {
  Activity,
  Archive,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronUp,
  Database,
  Download,
  ExternalLink,
  Layers3,
  ListChecks,
  LogOut,
  Play,
  Plus,
  RefreshCw,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { type FormEvent, type ReactNode, useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, apiDownload, apiForm, apiJson, downloadBlob } from "./api/client";
import type {
  AnyTemplate,
  AutomationRuleInfo,
  AutomationSnapshot,
  BootstrapInfo,
  ClassInfo,
  ClassResultsResponse,
  ConfigField,
  ConfigSchema,
  CustomDatasetInfo,
  DatasetInfo,
  ImageFolderInfo,
  InferenceTemplate,
  JobDetail,
  JobLogInfo,
  JobSummary,
  JsonRecord,
  MLflowExperimentInfo,
  ModelInfo,
  PseudoMarkupResultInfo,
  QueueSnapshot,
  ResultChangeInfo,
  ResultChangesResponse,
  TrainingResultInfo,
  TrainingTemplate,
} from "./api/types";
import {
  displayStoredFileName,
  exportModelNamePart,
  formatDate,
  formatDateTime,
  formatF1Score,
  formatFileSize,
  formatGeojsonSummary,
  formatRuntimeMinutes,
  integerOrNull,
  isValidExportModelName,
  runningProgressLabel,
  shortVersion,
} from "./utils/format";

const PROGRESS_REFRESH_MS = 10_000;

type ModalState = {
  title: string;
  body: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
};

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;

export function App() {
  const [route, setRoute] = useState(currentRoute());
  const [user, setUser] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [bootstrap, setBootstrap] = useState<BootstrapInfo | null>(null);
  const [modal, setModal] = useState<ModalState | null>(null);

  const closeModal = useCallback(() => setModal(null), []);

  const run = useCallback<Runner>(async (operation) => {
    try {
      return await operation();
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setUser(null);
        setBootstrap(null);
        return undefined;
      }
      setModal({
        title: "Ошибка",
        body: <p>{error instanceof Error ? error.message : "Неизвестная ошибка"}</p>,
      });
      return undefined;
    }
  }, []);

  const loadBootstrap = useCallback(async () => {
    const payload = await run(() => apiJson<BootstrapInfo>("/bootstrap"));
    if (payload) {
      setBootstrap(payload);
    }
  }, [run]);

  useEffect(() => {
    const onHashChange = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    apiJson<{ authenticated: boolean; username: string | null }>("/auth/me", { authOptional: true })
      .then((payload) => {
        if (cancelled) return;
        setUser(payload?.authenticated ? payload.username || "" : null);
      })
      .catch(() => {
        if (!cancelled) setUser(null);
      })
      .finally(() => {
        if (!cancelled) setAuthChecked(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (user && !bootstrap) {
      void loadBootstrap();
    }
  }, [bootstrap, loadBootstrap, user]);

  const logout = async () => {
    await run(() => apiJson<{ status: string }>("/auth/logout", { method: "POST" }));
    setUser(null);
    setBootstrap(null);
  };

  const showJobLog = useCallback(
    async (jobId: string) => {
      const log = await run(() => apiJson<JobLogInfo>(`/jobs/${encodeURIComponent(jobId)}/log`));
      if (!log) return;
      setModal({
        title: `Лог задания ${jobId}`,
        wide: true,
        body: (
          <div className="form-stack">
            <div className="inline-row">
              <span className="badge neutral">{log.source_name}</span>
              <span className="badge neutral">{formatFileSize(log.size_bytes)}</span>
              {log.truncated ? <span className="badge warning">показан хвост файла</span> : null}
            </div>
            <pre className="log-view">{log.content || "Лог пуст"}</pre>
          </div>
        ),
      });
    },
    [run],
  );

  if (!authChecked) {
    return <LoadingPage text="Проверка сессии" />;
  }

  if (!user) {
    return <LoginPage onLogin={setUser} run={run} />;
  }

  const page = bootstrap ? (
    <RoutedPage
      route={route}
      bootstrap={bootstrap}
      run={run}
      reloadBootstrap={loadBootstrap}
      showModal={setModal}
      closeModal={closeModal}
      showJobLog={showJobLog}
    />
  ) : (
    <LoadingPage text="Загрузка справочников" />
  );

  return (
    <Shell user={user} route={route} onLogout={logout}>
      {page}
      <Modal modal={modal} onClose={closeModal} />
    </Shell>
  );
}

function RoutedPage(props: {
  route: string[];
  bootstrap: BootstrapInfo;
  run: Runner;
  reloadBootstrap: () => Promise<void>;
  showModal: (modal: ModalState) => void;
  closeModal: () => void;
  showJobLog: (jobId: string) => Promise<void>;
}) {
  const [head, second] = props.route;
  if (head === "start") return <StartPage {...props} />;
  if (head === "queue") return <QueuePage {...props} />;
  if (head === "templates") return <TemplatesPage {...props} />;
  if (head === "automation") return <AutomationPage {...props} />;
  if (head === "model-export") return <ModelExportPage {...props} />;
  if (head === "results" && second) return <ClassResultsPage {...props} classKey={decodeURIComponent(second)} />;
  if (head === "results") return <ResultsPage {...props} />;
  if (head === "jobs" && second) return <JobPage {...props} jobId={second} />;
  return <HomePage {...props} />;
}

function Shell({
  user,
  route,
  onLogout,
  children,
}: {
  user: string;
  route: string[];
  onLogout: () => void;
  children: ReactNode;
}) {
  const navItems = [
    { href: "#/start", key: "start", label: "Запуск", icon: Play },
    { href: "#/queue", key: "queue", label: "Очередь", icon: ListChecks },
    { href: "#/templates", key: "templates", label: "Шаблоны", icon: Settings },
    { href: "#/automation", key: "automation", label: "Автоматизация", icon: Activity },
    { href: "#/model-export", key: "model-export", label: "Zip", icon: Archive },
    { href: "#/results", key: "results", label: "Результаты", icon: BarChart3 },
  ];
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#/">
          <BrainCircuit size={22} />
          MLSystem2
        </a>
        <nav className="nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <a className={route[0] === item.key ? "active" : ""} href={item.href} key={item.key}>
                <Icon size={16} />
                {item.label}
              </a>
            );
          })}
          <button type="button" title={`Выйти: ${user}`} onClick={onLogout}>
            <LogOut size={16} />
            Выйти
          </button>
        </nav>
      </header>
      <main className="page">{children}</main>
    </div>
  );
}

function LoginPage({ onLogin, run }: { onLogin: (user: string) => void; run: Runner }) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    const data = new FormData(event.currentTarget);
    const username = String(data.get("username") || "");
    const result = await run(() =>
      apiJson<{ status: string }>("/auth/login", {
        method: "POST",
        body: { username, password: String(data.get("password") || "") },
      }),
    );
    setBusy(false);
    if (result) onLogin(username);
    else setError("Неверный логин или пароль");
  };

  return (
    <main className="login-page">
      <section className="login-panel">
        <div className="login-mark">
          <BrainCircuit size={24} />
          MLSystem2
        </div>
        <form className="form-stack" onSubmit={submit}>
          <label className="field">
            <span>Логин</span>
            <input name="username" autoComplete="username" required />
          </label>
          <label className="field">
            <span>Пароль</span>
            <input name="password" type="password" autoComplete="current-password" required />
          </label>
          {error ? <div className="error-box">{error}</div> : null}
          <button className="primary" type="submit" disabled={busy}>
            <Check size={16} />
            Войти
          </button>
        </form>
      </section>
    </main>
  );
}

function LoadingPage({ text }: { text: string }) {
  return (
    <section className="panel">
      <div className="inline-row">
        <RefreshCw size={16} />
        <span>{text}</span>
      </div>
    </section>
  );
}

function HomePage({ bootstrap, run, showJobLog }: RoutedPageProps) {
  const [changes, setChanges] = useState<ResultChangeInfo[]>([]);
  const links = useMemo(() => Object.fromEntries(bootstrap.links.map((item) => [item.key, item])), [bootstrap.links]);

  useEffect(() => {
    void run(() => apiJson<ResultChangesResponse>("/results/changes")).then((payload) => {
      if (payload) setChanges(payload.changes || []);
    });
  }, [run]);

  return (
    <>
      <PageHeader title="Рабочая панель" subtitle="Обучение, очереди, результаты и сервисные ссылки MLSystem2" />
      <section className="content-grid">
        <ToolCard link={links.grafana} fallbackTitle="Grafana" icon={<BarChart3 size={20} />} />
        <ToolCard link={links.mlflow} fallbackTitle="MLflow" icon={<Activity size={20} />} />
        <ToolCard link={links.minio} fallbackTitle="MinIO" icon={<Database size={20} />} />
        <a className="tool-card" href="#/start">
          <div>
            <div className="card-title">
              <Play size={20} />
              Запуск обучения
            </div>
            <p className="muted">Создать training job из текущих шаблонов.</p>
          </div>
          <span className="secondary compact-action">Открыть</span>
        </a>
      </section>
      <section className="panel">
        <PanelHeader title="Последние изменения" subtitle="Training и pseudo-markup события" />
        <ResultChangesTable changes={changes} showJobLog={showJobLog} />
      </section>
    </>
  );
}

function ToolCard({ link, fallbackTitle, icon }: { link?: { title: string; url: string }; fallbackTitle: string; icon: ReactNode }) {
  return (
    <a className="tool-card" href={link?.url || "#"} target={link?.url ? "_blank" : undefined} rel="noreferrer">
      <div>
        <div className="card-title">
          {icon}
          {link?.title || fallbackTitle}
        </div>
        <p className="muted">{link?.url ? link.url : "Ссылка не настроена"}</p>
      </div>
      <span className="secondary compact-action">
        <ExternalLink size={14} />
        Открыть
      </span>
    </a>
  );
}

function StartPage({ bootstrap, run, reloadBootstrap, showModal, closeModal }: RoutedPageProps) {
  const [experiments, setExperiments] = useState<MLflowExperimentInfo[]>([]);
  const [architecture, setArchitecture] = useState(bootstrap.models[0]?.architecture || "");
  const [datasetKey, setDatasetKey] = useState(bootstrap.datasets[0]?.key || "");
  const [experimentId, setExperimentId] = useState("");
  const [experimentName, setExperimentName] = useState("MLSystem2");
  const [config, setConfig] = useState<JsonRecord>({});
  const [busy, setBusy] = useState(false);

  const template = useMemo(
    () => templateFor(bootstrap.training_templates, architecture, datasetKey),
    [architecture, bootstrap.training_templates, datasetKey],
  );

  useEffect(() => {
    void run(() => apiJson<MLflowExperimentInfo[]>("/mlflow/experiments")).then((payload) => {
      const list = payload || [];
      setExperiments(list);
      setExperimentId(latestExperimentId(list));
    });
  }, [run]);

  useEffect(() => {
    setConfig({ ...(template?.default_config || {}) });
  }, [template?.id]);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!template) return;
    setBusy(true);
    const formData = new FormData(event.currentTarget);
    let customDatasetId: string | null = null;
    if (datasetKey === "custom") {
      const scenesFile = formData.get("scenes_txt");
      const geojsonFile = formData.get("annotation_geojson");
      if (!(scenesFile instanceof File) || !scenesFile.name || !(geojsonFile instanceof File) || !geojsonFile.name) {
        showModal({ title: "Ошибка", body: <p>Для Custom нужны GeoJSON и TXT со снимками.</p> });
        setBusy(false);
        return;
      }
      const customForm = new FormData();
      customForm.set("name", "Custom");
      customForm.set("scenes_txt", scenesFile);
      customForm.set("annotation_geojson", geojsonFile);
      const custom = await run(() => apiForm<CustomDatasetInfo>("/custom-datasets", customForm));
      if (!custom) {
        setBusy(false);
        return;
      }
      customDatasetId = custom.id;
    }

    let selectedExperimentId = experimentId || null;
    let selectedExperimentName = experimentName.trim() || "MLSystem2";
    if (!selectedExperimentId) {
      const created = await run(() =>
        apiJson<MLflowExperimentInfo>("/mlflow/experiments", {
          method: "POST",
          body: { name: selectedExperimentName },
        }),
      );
      if (!created) {
        setBusy(false);
        return;
      }
      selectedExperimentId = created.experiment_id;
      selectedExperimentName = created.name;
    } else {
      selectedExperimentName =
        experiments.find((item) => item.experiment_id === selectedExperimentId)?.name || selectedExperimentName;
    }

    const created = await run(() =>
      apiJson<JobDetail>("/training-jobs", {
        method: "POST",
        body: {
          mlflow_experiment_id: selectedExperimentId,
          mlflow_experiment_name: selectedExperimentName,
          dataset_key: datasetKey,
          custom_dataset_id: customDatasetId,
          architecture,
          config,
        },
      }),
    );
    setBusy(false);
    if (!created) return;
    showModal({
      title: "Обучение запущено",
      body: <p>Задание добавлено в очередь обучения.</p>,
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>
            Закрыть
          </button>
          <a className="primary" href={`#/jobs/${created.id}`} onClick={closeModal}>
            Открыть job
          </a>
        </>
      ),
    });
    await reloadBootstrap();
  };

  return (
    <>
      <PageHeader title="Запуск обучения" subtitle="Создание training job в очереди MLSystem2" />
      <form className="form-stack" onSubmit={submit}>
        <section className="panel">
          <div className="form-grid">
            <label className="field">
              <span>MLflow experiment</span>
              <select value={experimentId} onChange={(event) => setExperimentId(event.target.value)}>
                <option value="">Новый experiment</option>
                {experiments.map((item) => (
                  <option value={item.experiment_id} key={item.experiment_id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            {!experimentId ? (
              <label className="field">
                <span>Новое имя experiment</span>
                <input value={experimentName} onChange={(event) => setExperimentName(event.target.value)} />
              </label>
            ) : null}
            <label className="field">
              <span>Датасет</span>
              <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)}>
                {bootstrap.datasets.map((item) => (
                  <option value={item.key} key={item.key}>
                    {datasetOptionLabel(item)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Модель</span>
              <select value={architecture} onChange={(event) => setArchitecture(event.target.value)}>
                {bootstrap.models.map((item) => (
                  <option value={item.architecture} key={item.architecture}>
                    {item.display_name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>
        {datasetKey === "custom" ? (
          <section className="panel">
            <PanelHeader title="Custom dataset" subtitle="Разовая загрузка GeoJSON и TXT со списком снимков" />
            <div className="form-grid">
              <label className="field">
                <span>GeoJSON</span>
                <input name="annotation_geojson" type="file" accept=".geojson,application/geo+json" />
              </label>
              <label className="field">
                <span>TXT со снимками</span>
                <input name="scenes_txt" type="file" accept=".txt,text/plain" />
              </label>
            </div>
          </section>
        ) : null}
        <section className="panel">
          <PanelHeader
            title="Параметры"
            subtitle={template ? template.display_name : "Шаблон не найден"}
            aside={template ? <span className="badge neutral">version={template.version}</span> : null}
          />
          {template ? (
            <ConfigEditor schema={template.config_schema} value={config} onChange={setConfig} />
          ) : (
            <div className="error-box">Нет шаблона для выбранной модели.</div>
          )}
        </section>
        <div className="button-row">
          <button className="primary" type="submit" disabled={busy || !template}>
            <Play size={16} />
            Запустить обучение
          </button>
        </div>
      </form>
    </>
  );
}

function ModelExportPage({ showModal, closeModal }: RoutedPageProps) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const modelName = String(data.get("model_name") || "").trim();
    const checkpoint = data.get("checkpoint");
    if (!isValidExportModelName(modelName)) {
      showModal({ title: "Ошибка", body: <p>Имя модели должно содержать только a-z, 0-9, дефис и подчеркивание.</p> });
      return;
    }
    if (!(checkpoint instanceof File) || !checkpoint.name.toLowerCase().endsWith(".pt")) {
      showModal({ title: "Ошибка", body: <p>Выберите MLSystem2 checkpoint .pt.</p> });
      return;
    }
    await exportCheckpointArchive(modelName, checkpoint, null, setBusy, setStatus, showModal, closeModal);
  };

  return (
    <>
      <PageHeader title="Экспорт модели" subtitle="Сборка zip-архива для models-serving-service и текущего Triton" />
      <form className="form-stack" onSubmit={submit}>
        <section className="panel">
          <div className="form-grid">
            <label className="field">
              <span>Имя модели</span>
              <input name="model_name" pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?" placeholder="deforestation-b2" required />
            </label>
            <label className="field">
              <span>Checkpoint .pt</span>
              <input name="checkpoint" type="file" accept=".pt" required />
            </label>
          </div>
        </section>
        <div className="inline-row">
          <button className="primary" type="submit" disabled={busy}>
            <Archive size={16} />
            Собрать zip
          </button>
          {status ? <span className="info-box">{status}</span> : null}
        </div>
      </form>
    </>
  );
}

function TemplatesPage({ bootstrap, run, reloadBootstrap, showModal, closeModal }: RoutedPageProps) {
  const [trainingId, setTrainingId] = useState(bootstrap.training_templates[0]?.id || "");
  const [inferenceId, setInferenceId] = useState(bootstrap.inference_templates[0]?.id || "");
  const trainingTemplate = byId(bootstrap.training_templates, trainingId) || bootstrap.training_templates[0];
  const inferenceTemplate = byId(bootstrap.inference_templates, inferenceId) || bootstrap.inference_templates[0];
  const [trainingConfig, setTrainingConfig] = useState<JsonRecord>({});
  const [inferenceConfig, setInferenceConfig] = useState<JsonRecord>({});

  useEffect(() => setTrainingConfig({ ...(trainingTemplate?.default_config || {}) }), [trainingTemplate?.id]);
  useEffect(() => setInferenceConfig({ ...(inferenceTemplate?.default_config || {}) }), [inferenceTemplate?.id]);

  const saveTemplate = async (mode: "training" | "inference", template: AnyTemplate, config: JsonRecord) => {
    const path = mode === "training" ? "training-templates" : "inference-templates";
    const updated = await run(() =>
      apiJson<AnyTemplate>(`/${path}/by-id/${template.id}`, {
        method: "PUT",
        body: { default_config: config },
      }),
    );
    if (updated) await reloadBootstrap();
  };

  const resetTemplate = async (mode: "training" | "inference", template: AnyTemplate) => {
    const path = mode === "training" ? "training-templates" : "inference-templates";
    const updated = await run(() =>
      apiJson<AnyTemplate>(`/${path}/by-id/${template.id}`, {
        method: "PUT",
        body: { reset_to_baseline: true },
      }),
    );
    if (updated) await reloadBootstrap();
  };

  const deleteTemplate = (mode: "training" | "inference", template: AnyTemplate) => {
    const path = mode === "training" ? "training-templates" : "inference-templates";
    showModal({
      title: "Удалить шаблон",
      body: <p>{template.display_name}</p>,
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>
            Отмена
          </button>
          <button
            className="danger"
            type="button"
            onClick={async () => {
              const deleted = await run(() => apiJson<AnyTemplate>(`/${path}/by-id/${template.id}`, { method: "DELETE" }));
              if (deleted) {
                closeModal();
                await reloadBootstrap();
              }
            }}
          >
            <Trash2 size={16} />
            Удалить
          </button>
        </>
      ),
    });
  };

  const applyFieldToAll = async (mode: "training" | "inference", template: AnyTemplate, key: string, value: unknown) => {
    const path = mode === "training" ? "training-templates" : "inference-templates";
    const updated = await run(() =>
      apiJson<unknown>(`/${path}/by-id/${template.id}/apply-field-to-all`, {
        method: "PUT",
        body: { key, value },
      }),
    );
    if (updated) await reloadBootstrap();
  };

  const showCreateModal = (mode: "training" | "inference") => {
    const templates = mode === "training" ? bootstrap.training_templates : bootstrap.inference_templates;
    showModal({
      title: mode === "training" ? "Добавить шаблон обучения" : "Добавить шаблон инференса",
      body: (
        <CreateTemplateForm
          mode={mode}
          models={bootstrap.models}
          datasets={bootstrap.datasets}
          templates={templates}
          run={run}
          closeModal={closeModal}
          reloadBootstrap={reloadBootstrap}
        />
      ),
    });
  };

  return (
    <>
      <PageHeader title="Шаблоны" subtitle="Базовые defaults сети и переопределения для конкретных датасетов" />
      <section className="two-column">
        <div className="form-stack">
          <TemplateTree
            title="Шаблоны обучения"
            templates={bootstrap.training_templates}
            selectedId={trainingTemplate?.id || ""}
            onSelect={setTrainingId}
            onAdd={() => showCreateModal("training")}
          />
          <TemplateTree
            title="Шаблоны инференса"
            templates={bootstrap.inference_templates}
            selectedId={inferenceTemplate?.id || ""}
            onSelect={setInferenceId}
            onAdd={() => showCreateModal("inference")}
          />
        </div>
        <div className="form-stack">
          {trainingTemplate ? (
            <TemplateEditor
              mode="training"
              template={trainingTemplate}
              config={trainingConfig}
              onConfig={setTrainingConfig}
              onSave={() => saveTemplate("training", trainingTemplate, trainingConfig)}
              onReset={() => resetTemplate("training", trainingTemplate)}
              onDelete={trainingTemplate.dataset_key ? () => deleteTemplate("training", trainingTemplate) : undefined}
              onApplyField={(key, value) => applyFieldToAll("training", trainingTemplate, key, value)}
            />
          ) : null}
          {inferenceTemplate ? (
            <TemplateEditor
              mode="inference"
              template={inferenceTemplate}
              config={inferenceConfig}
              onConfig={setInferenceConfig}
              onSave={() => saveTemplate("inference", inferenceTemplate, inferenceConfig)}
              onReset={() => resetTemplate("inference", inferenceTemplate)}
              onDelete={inferenceTemplate.dataset_key ? () => deleteTemplate("inference", inferenceTemplate) : undefined}
              onApplyField={(key, value) => applyFieldToAll("inference", inferenceTemplate, key, value)}
            />
          ) : null}
        </div>
      </section>
    </>
  );
}

function AutomationPage({ run, showModal, closeModal }: RoutedPageProps) {
  const [snapshot, setSnapshot] = useState<AutomationSnapshot | null>(null);
  const load = useCallback(async () => {
    const payload = await run(() => apiJson<AutomationSnapshot>("/automation"));
    if (payload) setSnapshot(payload);
  }, [run]);

  useEffect(() => {
    void load();
  }, [load]);

  if (!snapshot) return <LoadingPage text="Загрузка automation matrix" />;
  const rules = new Map(snapshot.rules.map((rule) => [automationRuleKey(rule.dataset_key, rule.architecture), rule]));

  const setEnabled = async (enabled: boolean) => {
    const updated = await run(() => apiJson<AutomationSnapshot>("/automation/enabled", { method: "PUT", body: { enabled } }));
    if (updated) setSnapshot(updated);
  };

  const toggleRule = async (dataset: DatasetInfo, model: ModelInfo, kind: "training" | "pseudo") => {
    const current = rules.get(automationRuleKey(dataset.key, model.architecture));
    const body = {
      dataset_key: dataset.key,
      architecture: model.architecture,
      training_enabled: kind === "training" ? !current?.training_enabled : Boolean(current?.training_enabled),
      pseudo_markup_enabled: kind === "pseudo" ? !current?.pseudo_markup_enabled : Boolean(current?.pseudo_markup_enabled),
    };
    const updated = await run(() => apiJson<AutomationRuleInfo>("/automation/rules", { method: "PUT", body }));
    if (updated) await load();
  };

  return (
    <>
      <PageHeader
        title="Автоматизация"
        subtitle="Правила запуска обучения и pseudo-markup при обновлении датасетов"
        actions={
          snapshot.enabled ? (
            <button
              className="danger"
              type="button"
              onClick={() =>
                showModal({
                  title: "Отключить автоматизацию",
                  body: <p>Новые задания по матрице перестанут создаваться.</p>,
                  footer: (
                    <>
                      <button className="secondary" type="button" onClick={closeModal}>
                        Отмена
                      </button>
                      <button
                        className="danger"
                        type="button"
                        onClick={async () => {
                          await setEnabled(false);
                          closeModal();
                        }}
                      >
                        Отключить
                      </button>
                    </>
                  ),
                })
              }
            >
              Отключить
            </button>
          ) : (
            <button className="primary" type="button" onClick={() => setEnabled(true)}>
              Включить
            </button>
          )
        }
      />
      <section className="panel">
        <div className="table-wrap">
          <table className="automation-table">
            <thead>
              <tr>
                <th>Датасет</th>
                {snapshot.models.map((model) => (
                  <th key={model.architecture}>{model.display_name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {snapshot.datasets.map((dataset) => (
                <tr key={dataset.key}>
                  <td>
                    <strong>{dataset.name}</strong>
                    <div className="muted">{shortVersion(dataset.version)}</div>
                  </td>
                  {snapshot.models.map((model) => {
                    const rule = rules.get(automationRuleKey(dataset.key, model.architecture));
                    return (
                      <td key={model.architecture}>
                        <div className="automation-cell">
                          <button
                            className={`automation-toggle ${rule?.training_enabled ? "enabled" : ""}`}
                            type="button"
                            onClick={() => toggleRule(dataset, model, "training")}
                          >
                            train {statusTiny(rule?.training_status)}
                          </button>
                          <button
                            className={`automation-toggle ${rule?.pseudo_markup_enabled ? "enabled" : ""}`}
                            type="button"
                            onClick={() => toggleRule(dataset, model, "pseudo")}
                          >
                            pseudo {statusTiny(rule?.pseudo_markup_status)}
                          </button>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function QueuePage({ run }: RoutedPageProps) {
  const [snapshot, setSnapshot] = useState<QueueSnapshot | null>(null);
  const load = useCallback(async () => {
    const payload = await run(() => apiJson<QueueSnapshot>("/queues"));
    if (payload) setSnapshot(payload);
  }, [run]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!snapshot || !(snapshot.jobs || []).some((job) => isActiveStatus(job.status))) return undefined;
    const timer = window.setTimeout(() => void load(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [load, snapshot]);

  const updateEnabled = async (queue: "training" | "inference", enabled: boolean) => {
    const updated = await run(() =>
      apiJson<QueueSnapshot>(`/queues/${queue}/enabled`, {
        method: "PUT",
        body: { enabled },
      }),
    );
    if (updated) setSnapshot(updated);
  };

  const jobAction = async (job: JobSummary, action: "move-up" | "move-down" | "delete") => {
    const path = action === "delete" ? `/jobs/${job.id}` : `/jobs/${job.id}/${action}`;
    const method = action === "delete" ? "DELETE" : "POST";
    const updated = await run(() => apiJson<JobDetail>(path, { method }));
    if (updated) await load();
  };

  if (!snapshot) return <LoadingPage text="Загрузка очереди" />;

  return (
    <>
      <PageHeader
        title="Очередь"
        subtitle="Запланированные и выполняющиеся training/inference задания"
        actions={
          <div className="inline-row">
            <button className={snapshot.training_enabled ? "primary" : "secondary"} type="button" onClick={() => updateEnabled("training", !snapshot.training_enabled)}>
              Training {snapshot.training_enabled ? "on" : "off"}
            </button>
            <button className={snapshot.inference_enabled ? "primary" : "secondary"} type="button" onClick={() => updateEnabled("inference", !snapshot.inference_enabled)}>
              Inference {snapshot.inference_enabled ? "on" : "off"}
            </button>
            <button className="secondary icon-button" type="button" onClick={load} title="Обновить">
              <RefreshCw size={16} />
            </button>
          </div>
        }
      />
      <section className="panel">
        <QueueTable jobs={snapshot.jobs || mergedQueueJobs(snapshot)} onAction={jobAction} />
      </section>
    </>
  );
}

function JobPage({ bootstrap, run, jobId }: RoutedPageProps & { jobId: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const load = useCallback(async () => {
    const payload = await run(() => apiJson<JobDetail>(`/jobs/${encodeURIComponent(jobId)}`));
    if (payload) setJob(payload);
  }, [jobId, run]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!job || !isActiveStatus(job.status)) return undefined;
    const timer = window.setTimeout(() => void load(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [job, load]);

  if (!job) return <LoadingPage text="Загрузка job detail" />;

  return (
    <>
      <PageHeader
        title={`Job ${job.id}`}
        subtitle={`${job.dataset_name} · ${job.model_name}`}
        actions={<a className="secondary" href="#/queue">К очереди</a>}
      />
      <section className="panel">
        <div className="metric-grid">
          <Metric label="Статус" value={statusBadge(job.status, job.type, job.progress)} />
          <Metric label="Тип" value={job.type} />
          <Metric label="Источник" value={sourceBadge(job.source)} />
          <Metric label="Создано" value={formatDateTime(job.created_at)} />
          <Metric label="Старт" value={formatDateTime(job.started_at)} />
          <Metric label="Финиш" value={formatDateTime(job.finished_at)} />
        </div>
      </section>
      <section className="panel">
        <PanelHeader title="Конфиг" subtitle={job.readonly ? "readonly snapshot" : ""} />
        <div className="job-config">
          {Object.entries(job.config || {}).map(([key, value]) => {
            const tooltip = configTooltipForKey(bootstrap, key);
            return (
              <div className="metric config-metric" key={key} title={tooltip || key}>
                <span className="muted">{key}:</span>
                <code>{formatConfigValue(value)}</code>
              </div>
            );
          })}
        </div>
      </section>
    </>
  );
}

function ResultsPage({ bootstrap, run, showJobLog }: RoutedPageProps) {
  const [changes, setChanges] = useState<ResultChangeInfo[]>([]);
  useEffect(() => {
    void run(() => apiJson<ResultChangesResponse>("/results/changes")).then((payload) => {
      if (payload) setChanges(payload.changes || []);
    });
  }, [run]);

  return (
    <>
      <PageHeader title="Результаты" subtitle="Классы, варианты датасетов и последние изменения" />
      <section className="content-grid">
        {bootstrap.classes.map((item) => (
          <ResultClassCard item={item} key={item.key} />
        ))}
      </section>
      <section className="panel">
        <PanelHeader title="Последние изменения" />
        <ResultChangesTable changes={changes} showJobLog={showJobLog} />
      </section>
    </>
  );
}

function ClassResultsPage({
  classKey,
  bootstrap,
  run,
  showModal,
  closeModal,
  showJobLog,
}: RoutedPageProps & { classKey: string }) {
  const [payload, setPayload] = useState<ClassResultsResponse | null>(null);
  const load = useCallback(async () => {
    const data = await run(() => apiJson<ClassResultsResponse>(`/results/classes/${encodeURIComponent(classKey)}`));
    if (data) setPayload(data);
  }, [classKey, run]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!payload || !hasActiveClassResults(payload)) return undefined;
    const timer = window.setTimeout(() => void load(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [load, payload]);

  if (!payload) return <LoadingPage text="Загрузка результатов класса" />;

  const showPseudo = (result: TrainingResultInfo) => {
    showModal({
      title: "Запустить pseudo-markup",
      body: (
        <PseudoMarkupForm
          classKey={classKey}
          result={result}
          datasets={bootstrap.datasets}
          imageFolders={bootstrap.image_folders}
          run={run}
          closeModal={closeModal}
          reload={load}
        />
      ),
    });
  };

  const showZip = (result: TrainingResultInfo) => {
    showTrainingResultZipModal(result, bootstrap.datasets, run, showModal, closeModal);
  };

  const deletePseudo = (item: PseudoMarkupResultInfo) => {
    showModal({
      title: "Удалить pseudo-markup",
      body: <p>{imageSourceLabel(item, bootstrap.datasets, bootstrap.image_folders)}</p>,
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>
            Отмена
          </button>
          <button
            className="danger"
            type="button"
            onClick={async () => {
              const deleted = await run(() => apiJson<PseudoMarkupResultInfo>(`/results/pseudo-markup/${item.id}`, { method: "DELETE" }));
              if (deleted) {
                closeModal();
                await load();
              }
            }}
          >
            <Trash2 size={16} />
            Удалить
          </button>
        </>
      ),
    });
  };

  return (
    <>
      <PageHeader
        title={payload.class_name}
        subtitle={`Обновление датасета: ${formatDate(payload.dataset_updated_at)}`}
        actions={<a className="secondary" href="#/results">Все классы</a>}
      />
      <section className="panel">
        <ResultsTable
          payload={payload}
          datasets={bootstrap.datasets}
          imageFolders={bootstrap.image_folders}
          onPseudo={showPseudo}
          onZip={showZip}
          onDeletePseudo={deletePseudo}
          showJobLog={showJobLog}
        />
      </section>
    </>
  );
}

type RoutedPageProps = {
  route: string[];
  bootstrap: BootstrapInfo;
  run: Runner;
  reloadBootstrap: () => Promise<void>;
  showModal: (modal: ModalState) => void;
  closeModal: () => void;
  showJobLog: (jobId: string) => Promise<void>;
};

function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div className="page-title">
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {actions ? <div className="button-row">{actions}</div> : null}
    </header>
  );
}

function PanelHeader({ title, subtitle, aside }: { title: string; subtitle?: string; aside?: ReactNode }) {
  return (
    <div className="panel-header">
      <div>
        <h2>{title}</h2>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {aside}
    </div>
  );
}

function ConfigEditor({
  schema,
  value,
  onChange,
  readonly = false,
  onApplyField,
}: {
  schema: ConfigSchema;
  value: JsonRecord;
  onChange: (next: JsonRecord) => void;
  readonly?: boolean;
  onApplyField?: (key: string, value: unknown) => void;
}) {
  const setField = (field: ConfigField, raw: unknown) => {
    const nextValue = coerceConfigValue(field, raw);
    onChange({ ...value, [field.key]: nextValue });
  };
  return (
    <div className="config-grid">
      {(schema.fields || []).map((field) => {
        const current = value[field.key] ?? "";
        const tooltip = configFieldTooltip(field);
        const label = (
          <span title={tooltip || field.label}>
            <span>{field.label}</span>
          </span>
        );
        if (field.value_type === "boolean") {
          return (
            <label className="field checkbox-field" key={field.key} title={tooltip || field.label}>
              <input
                type="checkbox"
                checked={Boolean(current)}
                disabled={readonly}
                onChange={(event) => setField(field, event.target.checked)}
              />
              <span>{field.label}</span>
              {onApplyField && !readonly ? (
                <button className="secondary compact-action" type="button" onClick={() => onApplyField(field.key, Boolean(current))}>
                  ко всем
                </button>
              ) : null}
            </label>
          );
        }
        return (
          <label className="field" key={field.key}>
            {label}
            <div className="inline-row">
              {field.options?.length ? (
                <select
                  value={String(current)}
                  disabled={readonly}
                  title={tooltip || field.label}
                  onChange={(event) => setField(field, event.target.value)}
                >
                  {field.options.map((option) => (
                    <option value={option} key={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={field.value_type.startsWith("integer") || field.value_type.startsWith("number") ? "number" : "text"}
                  step={field.value_type.startsWith("number") ? "any" : "1"}
                  value={String(current)}
                  disabled={readonly}
                  required={field.required}
                  onChange={(event) => setField(field, event.target.value)}
                  title={tooltip || field.label}
                />
              )}
              {onApplyField && !readonly ? (
                <button className="secondary compact-action" type="button" onClick={() => onApplyField(field.key, value[field.key])}>
                  ко всем
                </button>
              ) : null}
            </div>
          </label>
        );
      })}
    </div>
  );
}

function TemplateTree({
  title,
  templates,
  selectedId,
  onSelect,
  onAdd,
}: {
  title: string;
  templates: AnyTemplate[];
  selectedId: string;
  onSelect: (id: string) => void;
  onAdd: () => void;
}) {
  const bases = templates.filter((item) => !item.dataset_key);
  return (
    <section className="panel">
      <PanelHeader
        title={title}
        aside={
          <button className="secondary compact-action" type="button" onClick={onAdd}>
            <Plus size={14} />
            Добавить
          </button>
        }
      />
      <div className="template-tree">
        {bases.map((base) => (
          <div key={base.id}>
            <TreeButton template={base} active={base.id === selectedId} onClick={() => onSelect(base.id)} />
            {templates
              .filter((item) => item.architecture === base.architecture && item.dataset_key)
              .map((child) => (
                <TreeButton child template={child} active={child.id === selectedId} onClick={() => onSelect(child.id)} key={child.id} />
              ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function TreeButton({ template, active, child = false, onClick }: { template: AnyTemplate; active: boolean; child?: boolean; onClick: () => void }) {
  return (
    <button className={`tree-button ${child ? "child" : ""} ${active ? "active" : ""}`} type="button" onClick={onClick}>
      <span>{templateTitle(template)}</span>
      <span className={`badge ${template.source === "manual" ? "warning" : "ok"}`}>{template.source}</span>
    </button>
  );
}

function TemplateEditor({
  mode,
  template,
  config,
  onConfig,
  onSave,
  onReset,
  onDelete,
  onApplyField,
}: {
  mode: "training" | "inference";
  template: AnyTemplate;
  config: JsonRecord;
  onConfig: (next: JsonRecord) => void;
  onSave: () => void;
  onReset: () => void;
  onDelete?: () => void;
  onApplyField: (key: string, value: unknown) => void;
}) {
  return (
    <section className="panel">
      <PanelHeader
        title={`${mode === "training" ? "Training" : "Inference"}: ${templateTitle(template)}`}
        subtitle={template.dataset_key ? "Шаблон датасета" : "Базовый шаблон сети"}
        aside={<span className="badge neutral">version={template.version}</span>}
      />
      <ConfigEditor schema={template.config_schema} value={config} onChange={onConfig} onApplyField={onApplyField} />
      <div className="button-row">
        <button className="primary" type="button" onClick={onSave}>
          Сохранить
        </button>
        <button className="secondary" type="button" onClick={onReset}>
          Сбросить
        </button>
        {onDelete ? (
          <button className="danger" type="button" onClick={onDelete}>
            <Trash2 size={16} />
            Удалить
          </button>
        ) : null}
      </div>
    </section>
  );
}

function CreateTemplateForm({
  mode,
  models,
  datasets,
  templates,
  run,
  closeModal,
  reloadBootstrap,
}: {
  mode: "training" | "inference";
  models: ModelInfo[];
  datasets: DatasetInfo[];
  templates: AnyTemplate[];
  run: Runner;
  closeModal: () => void;
  reloadBootstrap: () => Promise<void>;
}) {
  const [architecture, setArchitecture] = useState(models[0]?.architecture || "");
  const availableDatasets = datasets.filter(
    (dataset) =>
      dataset.key !== "custom" &&
      !templates.some((template) => template.architecture === architecture && template.dataset_key === dataset.key),
  );
  const [datasetKey, setDatasetKey] = useState(availableDatasets[0]?.key || "");

  useEffect(() => {
    setDatasetKey(availableDatasets[0]?.key || "");
  }, [architecture]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const path = mode === "training" ? "training-templates" : "inference-templates";
    const created = await run(() =>
      apiJson<AnyTemplate>(`/${path}`, {
        method: "POST",
        body: { architecture, dataset_key: datasetKey },
      }),
    );
    if (created) {
      closeModal();
      await reloadBootstrap();
    }
  };

  return (
    <form className="form-stack" onSubmit={submit}>
      <label className="field">
        <span>Модель</span>
        <select value={architecture} onChange={(event) => setArchitecture(event.target.value)}>
          {models.map((model) => (
            <option value={model.architecture} key={model.architecture}>
              {model.display_name}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>Датасет</span>
        <select value={datasetKey} onChange={(event) => setDatasetKey(event.target.value)} required>
          {availableDatasets.map((dataset) => (
            <option value={dataset.key} key={dataset.key}>
              {dataset.name}
            </option>
          ))}
        </select>
      </label>
      <div className="button-row">
        <button className="primary" type="submit" disabled={!datasetKey}>
          Создать
        </button>
      </div>
    </form>
  );
}

function QueueTable({ jobs, onAction }: { jobs: JobSummary[]; onAction: (job: JobSummary, action: "move-up" | "move-down" | "delete") => void }) {
  if (!jobs.length) return <div className="empty-state">Очередь пуста</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Статус</th>
            <th>Тип</th>
            <th>Датасет</th>
            <th>Модель</th>
            <th>Создано</th>
            <th>Действия</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr className="clickable-row" key={job.id} onClick={() => navigate(`jobs/${job.id}`)}>
              <td>{job.queue_position}</td>
              <td>{statusBadge(job.status, job.type, job.progress)}</td>
              <td>{jobTypeBadge(job.type)}</td>
              <td>{queueDatasetCell(job)}</td>
              <td>{queueModelCell(job)}</td>
              <td>{formatDateTime(job.created_at)}</td>
              <td>
                <div className="inline-row" onClick={(event) => event.stopPropagation()}>
                  <button className="secondary icon-button" type="button" title="Выше" onClick={() => onAction(job, "move-up")}>
                    <ChevronUp size={15} />
                  </button>
                  <button className="secondary icon-button" type="button" title="Ниже" onClick={() => onAction(job, "move-down")}>
                    <ChevronDown size={15} />
                  </button>
                  <button className="danger icon-button" type="button" title="Удалить" onClick={() => onAction(job, "delete")}>
                    <Trash2 size={15} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultClassCard({ item }: { item: ClassInfo }) {
  const variants = item.variants?.length ? item.variants : [];
  return (
    <div className="class-card">
      <div className="card-title">
        <Layers3 size={20} />
        {item.name}
      </div>
      <p className="muted">Обновлено: {formatDate(item.updated_at)}</p>
      <div className="variant-list">
        {variants.length ? (
          variants.map((variant) => (
            <a className="variant-link" href={`#/results/${encodeURIComponent(variant.key)}`} key={variant.key}>
              <span>{variant.variant_name || variant.name}</span>
              <small>{integerOrNull(variant.image_count) ?? "—"} img</small>
            </a>
          ))
        ) : (
          <a className="variant-link" href={`#/results/${encodeURIComponent(item.key)}`}>
            <span>main</span>
            <small>открыть</small>
          </a>
        )}
      </div>
    </div>
  );
}

function ResultChangesTable({ changes, showJobLog }: { changes: ResultChangeInfo[]; showJobLog: (jobId: string) => Promise<void> }) {
  if (!changes.length) return <div className="empty-state">Изменений пока нет</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Статус</th>
            <th>Класс</th>
            <th>Датасет</th>
            <th>Модель</th>
            <th>Действие</th>
            <th>Время</th>
          </tr>
        </thead>
        <tbody>
          {changes.map((item) => (
            <tr
              className={`clickable-row result-change-row ${resultKindClass(changeResultKind(item))}`}
              key={item.id}
              onClick={() => navigate(`results/${encodeURIComponent(item.class_key)}`)}
            >
              <td onClick={(event) => event.stopPropagation()}>
                <span className="status-stack">
                  {resultStatusBadge(item.status, item.type, undefined, item.job_id, showJobLog)}
                  {sourceBadge(item.source)}
                </span>
              </td>
              <td>{item.class_key}</td>
              <td>{item.dataset_name}</td>
              <td>{item.model_name}</td>
              <td>{actionBadge(item.action, changeResultKind(item))}</td>
              <td>{formatDateTime(item.changed_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsTable({
  payload,
  datasets,
  imageFolders,
  onPseudo,
  onZip,
  onDeletePseudo,
  showJobLog,
}: {
  payload: ClassResultsResponse;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  onPseudo: (result: TrainingResultInfo) => void;
  onZip: (result: TrainingResultInfo) => void;
  onDeletePseudo: (item: PseudoMarkupResultInfo) => void;
  showJobLog: (jobId: string) => Promise<void>;
}) {
  if (!payload.results.length) return <div className="empty-state">Для класса пока нет результатов</div>;
  return (
    <div className="form-stack">
      {payload.results.map((result) => (
        <section className="result-group" key={result.id}>
          <div className="table-wrap">
            <table className="training-summary-table">
              <colgroup>
                <col className="result-col-model" />
                <col className="result-col-status" />
                <col className="result-col-score" />
                <col className="result-col-epoch" />
                <col className="result-col-created" />
                <col className="result-col-actions" />
              </colgroup>
              <thead className="visually-hidden-header">
                <tr>
                  <th>МОДЕЛЬ</th>
                  <th>Статус</th>
                  <th>F1</th>
                  <th>Epoch</th>
                  <th>Создано</th>
                  <th aria-label="Действия"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="training-result-row">
                  <td title="МОДЕЛЬ">
                    <span className="source-lines">
                      <strong>{result.model_name}</strong>
                      <small className="muted">{result.architecture}</small>
                    </span>
                  </td>
                  <td title="Статус">
                    <span className="status-stack">
                      {resultStatusBadge(result.status, "training", result.progress, result.job_id, showJobLog)}
                      {sourceBadge(result.source)}
                    </span>
                  </td>
                  <td title="F1">{formatF1Score(result.f1_score)}</td>
                  <td title="Epoch">{result.epoch ?? "—"}</td>
                  <td title="Создано">{formatDateTime(result.trained_at)}</td>
                  <td className="action-cell">
                    {result.status === "ok" ? (
                      <>
                        <button className="secondary compact-action" type="button" title="Запустить псевдоразметку" onClick={() => onPseudo(result)}>
                          <Play size={14} />
                          Pseudo
                        </button>
                        <button className="secondary compact-action" type="button" title="Скачать Triton zip" onClick={() => onZip(result)}>
                          <Archive size={14} />
                          Zip
                        </button>
                      </>
                    ) : null}
                    {result.mlflow_run_url ? (
                      <a className="secondary compact-action" href={result.mlflow_run_url} target="_blank" rel="noreferrer" title="Открыть MLflow run">
                        MLflow
                      </a>
                    ) : null}
                    {result.job_id ? (
                      <a className="secondary compact-action" href={`#/jobs/${result.job_id}`} title="Открыть job обучения">
                        Job
                      </a>
                    ) : null}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          {(result.pseudo_markup_results || []).length ? (
            <div className="table-wrap pseudo-subtable-wrap">
              <table className="pseudo-table">
                <colgroup>
                  <col className="result-col-source" />
                  <col className="result-col-status" />
                  <col className="result-col-geojson" />
                  <col className="result-col-created" />
                  <col className="result-col-actions" />
                </colgroup>
                <thead>
                  <tr>
                    <th>ИСТОЧНИК</th>
                    <th>Статус</th>
                    <th>GeoJSON</th>
                    <th>Создано</th>
                    <th aria-label="Действия"></th>
                  </tr>
                </thead>
                <tbody>
                  {(result.pseudo_markup_results || []).map((item) => (
                    <tr className="pseudo-result-row" key={item.id}>
                      <td title="ИСТОЧНИК">{imageSourceLabel(item, datasets, imageFolders)}</td>
                      <td title="Статус">
                        <span className="status-stack">
                          {resultStatusBadge(item.status, "inference", item.progress, item.job_id, showJobLog)}
                          {sourceBadge(item.source)}
                        </span>
                      </td>
                      <td title="GeoJSON">{item.geojson_file ? geojsonDownloadLink(item.geojson_file) : "—"}</td>
                      <td title="Создано">{pseudoCreatedLabel(item)}</td>
                      <td className="action-cell">
                        {item.job_id ? (
                          <a className="secondary compact-action" href={`#/jobs/${item.job_id}`} title="Открыть job разметки">
                            Job
                          </a>
                        ) : null}
                        <button className="danger icon-button" type="button" title="Удалить" onClick={() => onDeletePseudo(item)}>
                          <Trash2 size={15} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      ))}
    </div>
  );
}

function PseudoMarkupForm({
  classKey,
  result,
  datasets,
  imageFolders,
  run,
  closeModal,
  reload,
}: {
  classKey: string;
  result: TrainingResultInfo;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  run: Runner;
  closeModal: () => void;
  reload: () => Promise<void>;
}) {
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const source = new FormData(event.currentTarget);
    const datasetKey = String(source.get("dataset_key") || "");
    const imageFolderKey = String(source.get("image_folder_key") || "");
    const file = source.get("scenes_txt");
    const hasFile = file instanceof File && Boolean(file.name);
    const sourceCount = [Boolean(datasetKey), Boolean(imageFolderKey), hasFile].filter(Boolean).length;
    if (sourceCount !== 1) {
      window.alert("Выберите ровно один источник: датасет, папку снимков или TXT.");
      return;
    }
    const request = new FormData();
    request.set("training_result_id", result.id);
    if (datasetKey) request.set("dataset_key", datasetKey);
    if (imageFolderKey) request.set("image_folder_key", imageFolderKey);
    if (hasFile && file instanceof File) request.set("scenes_txt", file);
    const created = await run(() => apiForm<JobDetail>(`/results/classes/${encodeURIComponent(classKey)}/pseudo-markup`, request));
    if (created) {
      closeModal();
      await reload();
    }
  };
  return (
    <form className="form-stack" onSubmit={submit}>
      <label className="field">
        <span>Датасет</span>
        <select name="dataset_key" defaultValue="">
          <option value="">Не выбран</option>
          {datasets
            .filter((dataset) => dataset.key !== "custom")
            .map((dataset) => (
              <option value={dataset.key} key={dataset.key}>
                {datasetOptionLabel(dataset)}
              </option>
            ))}
        </select>
      </label>
      <label className="field">
        <span>Папка снимков</span>
        <select name="image_folder_key" defaultValue="">
          <option value="">Не выбрана</option>
          {imageFolders.map((folder) => (
            <option value={folder.key} key={folder.key}>
              {imageFolderOptionLabel(folder)}
            </option>
          ))}
        </select>
      </label>
      <label className="field">
        <span>TXT со снимками</span>
        <input name="scenes_txt" type="file" accept=".txt,text/plain" />
      </label>
      <button className="primary" type="submit">
        <Play size={16} />
        Запустить
      </button>
    </form>
  );
}

function Modal({ modal, onClose }: { modal: ModalState | null; onClose: () => void }) {
  if (!modal) return null;
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className={`modal-card ${modal.wide ? "wide" : ""}`} role="dialog" aria-modal="true">
        <header className="modal-header">
          <h2>{modal.title}</h2>
          <button className="ghost icon-button" type="button" onClick={onClose} aria-label="Закрыть">
            <X size={17} />
          </button>
        </header>
        <div className="modal-body">{modal.body}</div>
        <footer className="modal-footer">
          {modal.footer || (
            <button className="secondary" type="button" onClick={onClose}>
              Закрыть
            </button>
          )}
        </footer>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="metric">
      <span className="muted">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function currentRoute(): string[] {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return hash ? hash.split("/") : [];
}

function navigate(path: string) {
  window.location.hash = `#/${path.replace(/^#?\/?/, "")}`;
}

function byId<T extends { id: string }>(items: T[], id: string): T | undefined {
  return items.find((item) => item.id === id);
}

function templateFor(templates: TrainingTemplate[], architecture: string, datasetKey: string | null): TrainingTemplate | undefined {
  const datasetTemplate =
    datasetKey && datasetKey !== "custom"
      ? templates.find((item) => item.architecture === architecture && item.dataset_key === datasetKey && item.is_active)
      : undefined;
  return datasetTemplate || templates.find((item) => item.architecture === architecture && !item.dataset_key) || templates[0];
}

function templateTitle(template: AnyTemplate): string {
  return template.dataset_key ? `${template.display_name} · ${template.dataset_name || template.dataset_key}` : template.display_name;
}

function latestExperimentId(experiments: MLflowExperimentInfo[]): string {
  if (!experiments.length) return "";
  return experiments.reduce((latest, item) => {
    const latestNumber = Number(latest.experiment_id);
    const itemNumber = Number(item.experiment_id);
    if (Number.isFinite(latestNumber) && Number.isFinite(itemNumber)) {
      return itemNumber > latestNumber ? item : latest;
    }
    return item.name.localeCompare(latest.name) > 0 ? item : latest;
  }, experiments[0]).experiment_id;
}

function coerceConfigValue(field: ConfigField, raw: unknown): unknown {
  if (field.value_type === "boolean") return Boolean(raw);
  if (field.value_type.startsWith("integer")) {
    const number = Number.parseInt(String(raw), 10);
    return Number.isFinite(number) ? number : null;
  }
  if (field.value_type.startsWith("number")) {
    const number = Number.parseFloat(String(raw));
    return Number.isFinite(number) ? number : null;
  }
  return String(raw);
}

function configFieldTooltip(field: ConfigField): string {
  const parts = [field.tooltip, configAllowedRange(field), field.recommended_range ? `Рекомендуется: ${field.recommended_range}` : ""];
  return parts.filter(Boolean).join(" · ");
}

function configTooltipForKey(bootstrap: BootstrapInfo, key: string): string {
  const templates = [...bootstrap.training_templates, ...bootstrap.inference_templates];
  for (const template of templates) {
    const field = (template.config_schema.fields || []).find((candidate) => candidate.key === key);
    if (field) return configFieldTooltip(field);
  }
  return "";
}

function formatConfigValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function configAllowedRange(field: ConfigField): string {
  const min = field.min_value;
  const max = field.max_value;
  if (min !== null && min !== undefined && max !== null && max !== undefined) return `${min}..${max}`;
  if (min !== null && min !== undefined) return `>= ${min}`;
  if (max !== null && max !== undefined) return `<= ${max}`;
  return "";
}

function statusBadge(status: string, type?: string | null, progress?: { current?: number | null; total?: number | null; elapsed_minutes?: number | null } | null) {
  const className = statusClass(status);
  const label = status === "running" ? runningProgressLabel(type, progress || null) : statusLabel(status);
  return (
    <span className={`badge ${className}`}>
      {status === "running" ? <RefreshCw className="status-spinner" size={13} /> : null}
      {label}
    </span>
  );
}

function resultStatusBadge(
  status: string,
  type: string | null | undefined,
  progress: { current?: number | null; total?: number | null; elapsed_minutes?: number | null } | null | undefined,
  jobId: string | null | undefined,
  showJobLog: (jobId: string) => Promise<void>,
) {
  const badge = statusBadge(status, type, progress);
  if ((status === "error" || status === "failed") && jobId) {
    return (
      <button className="badge badge-button error" type="button" onClick={() => void showJobLog(jobId)}>
        {statusLabel(status)}
      </button>
    );
  }
  return badge;
}

type ResultKind = "training" | "pseudo" | "neutral";

function changeResultKind(item: ResultChangeInfo): ResultKind {
  if (item.item_type === "training_result" || item.type === "training") return "training";
  if (item.item_type === "pseudo_markup_result" || item.type === "inference") return "pseudo";
  const action = item.action.toLowerCase();
  if (action.includes("обуч")) return "training";
  if (action.includes("размет")) return "pseudo";
  return "neutral";
}

function resultKindClass(kind: ResultKind): string {
  return kind === "neutral" ? "" : `kind-${kind}`;
}

function actionBadge(action: string, kind: ResultKind) {
  return <span className={`badge action-badge ${resultKindClass(kind)}`}>{action}</span>;
}

function sourceBadge(source: string) {
  const automated = source === "automation";
  return <span className={`badge source-badge ${automated ? "auto" : "manual"}`}>{automated ? "auto" : "manual"}</span>;
}

function jobTypeBadge(type: string) {
  return <span className={`badge ${type === "inference" ? "warning" : "neutral"}`}>{type}</span>;
}

function statusClass(status: string): string {
  if (status === "ok" || status === "completed") return "ok";
  if (status === "queued" || status === "running") return status;
  if (status === "error" || status === "failed") return "error";
  if (status === "cancelled") return "warning";
  return "neutral";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: "в очереди",
    running: "в процессе",
    ok: "ok",
    completed: "завершено",
    error: "ошибка",
    failed: "ошибка",
    cancelled: "отменено",
  };
  return labels[status] || status;
}

function statusTiny(status?: string | null): string {
  return status ? `· ${statusLabel(status)}` : "";
}

function isActiveStatus(status: string): boolean {
  return status === "queued" || status === "running";
}

function hasActiveClassResults(payload: ClassResultsResponse): boolean {
  return payload.results.some(
    (item) => isActiveStatus(item.status) || (item.pseudo_markup_results || []).some((pseudo) => isActiveStatus(pseudo.status)),
  );
}

function datasetOptionLabel(item: DatasetInfo): string {
  const count = integerOrNull(item.image_count);
  return count === null ? item.name : `${item.name} (${count} img)`;
}

function imageFolderOptionLabel(item: ImageFolderInfo): string {
  return `${item.name} (${item.image_count} img)`;
}

function imageSourceLabel(item: PseudoMarkupResultInfo, datasets: DatasetInfo[], folders: ImageFolderInfo[]): string {
  const label = item.dataset_key
    ? datasets.find((dataset) => dataset.key === item.dataset_key)?.name || item.source_dataset_name
    : folders.find((candidate) => candidate.key === item.source_dataset_name)?.name || item.source_dataset_name;
  const count = integerOrNull(item.image_count);
  return `${label}${count === null ? "" : ` (${count} снимков)`}`;
}

function pseudoCreatedLabel(item: PseudoMarkupResultInfo): string {
  const runtime = formatRuntimeMinutes(item.runtime_minutes);
  return runtime ? `${formatDateTime(item.created_at)} (за ${runtime})` : formatDateTime(item.created_at);
}

function geojsonDownloadLink(file: { download_url: string; original_name: string; size_bytes: number; object_count?: number | null }) {
  const displayName = displayStoredFileName(file.original_name) || file.original_name;
  return (
    <a className="secondary compact-action file-download-link" href={file.download_url} title={displayName}>
      <Download size={14} />
      <span className="file-link-name">{formatGeojsonSummary(file.object_count, file.size_bytes)}</span>
    </a>
  );
}

function queueDatasetCell(job: JobSummary): ReactNode {
  if (job.type === "inference") {
    return (
      <span className="source-lines">
        <span>{job.inference_dataset_name || job.dataset_name}</span>
        {job.training_dataset_name ? <small className="muted">train: {job.training_dataset_name}</small> : null}
      </span>
    );
  }
  return job.dataset_name;
}

function queueModelCell(job: JobSummary): ReactNode {
  return (
    <span className="source-lines">
      <span>{job.model_name}</span>
      {job.tile_size ? <small className="muted">tile={job.tile_size}</small> : null}
    </span>
  );
}

function mergedQueueJobs(snapshot: QueueSnapshot): JobSummary[] {
  return [...(snapshot.training_jobs || []), ...(snapshot.inference_jobs || [])].sort((left, right) => {
    if (left.status !== right.status) return queuePriority(left) - queuePriority(right);
    return left.queue_position - right.queue_position;
  });
}

function queuePriority(job: JobSummary): number {
  if (job.status === "running") return 0;
  if (job.status === "queued") return 1;
  return 2;
}

function automationRuleKey(datasetKey: string, architecture: string): string {
  return `${datasetKey}::${architecture}`;
}

function defaultTrainingZipModelName(result: TrainingResultInfo, datasets: DatasetInfo[]): string {
  const dataset = datasets.find((item) => item.key === result.dataset_key);
  const filename = String(dataset?.annotation_file || "").split(/[\\/]/).pop() || "";
  const stem = filename.replace(/\.[^.]*$/, "");
  const datasetPart = exportModelNamePart(stem || dataset?.class_key || dataset?.name || result.dataset_key || "model");
  const suffix = "kanopus";
  if (!datasetPart || datasetPart === suffix || datasetPart.endsWith(`_${suffix}`)) return datasetPart || suffix;
  return `${datasetPart}_${suffix}`;
}

async function exportCheckpointArchive(
  modelName: string,
  checkpoint: File,
  sampleSize: number | null,
  setBusy: (busy: boolean) => void,
  setStatus: (status: string) => void,
  showModal: (modal: ModalState) => void,
  closeModal: () => void,
) {
  setBusy(true);
  setStatus("Сборка архива...");
  const request = new FormData();
  request.set("model_name", modelName);
  request.set("checkpoint", checkpoint);
  if (sampleSize !== null) request.set("sample_size", String(sampleSize));
  try {
    const response = await apiDownload("/model-export/triton-zip", request);
    downloadBlob(response.blob, response.filename || `${modelName}_export.zip`);
    setStatus("Архив готов");
  } catch (error) {
    if (error instanceof ApiError && error.message.includes("metadata.sample_size")) {
      showSampleSizeModal((value) => exportCheckpointArchive(modelName, checkpoint, value, setBusy, setStatus, showModal, closeModal), showModal, closeModal);
    } else {
      showModal({ title: "Ошибка экспорта", body: <p>{error instanceof Error ? error.message : "Неизвестная ошибка"}</p> });
    }
  } finally {
    setBusy(false);
  }
}

function showTrainingResultZipModal(
  result: TrainingResultInfo,
  datasets: DatasetInfo[],
  run: Runner,
  showModal: (modal: ModalState) => void,
  closeModal: () => void,
) {
  const defaultName = defaultTrainingZipModelName(result, datasets);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const modelName = String(data.get("model_name") || "").trim();
    if (!isValidExportModelName(modelName)) {
      window.alert("Имя модели должно содержать только a-z, 0-9, дефис и подчеркивание.");
      return;
    }
    await exportTrainingResultArchive(result.id, modelName, null, run, showModal, closeModal);
  };
  showModal({
    title: "Собрать Triton zip",
    body: (
      <form className="form-stack" onSubmit={submit}>
        <label className="field">
          <span>Имя модели</span>
          <input name="model_name" defaultValue={defaultName} pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?" required />
        </label>
        <button className="primary" type="submit">
          <Archive size={16} />
          Скачать zip
        </button>
      </form>
    ),
  });
}

async function exportTrainingResultArchive(
  resultId: string,
  modelName: string,
  sampleSize: number | null,
  run: Runner,
  showModal: (modal: ModalState) => void,
  closeModal: () => void,
) {
  const request = new FormData();
  request.set("model_name", modelName);
  if (sampleSize !== null) request.set("sample_size", String(sampleSize));
  try {
    const response = await apiDownload(`/results/training/${encodeURIComponent(resultId)}/triton-zip`, request);
    downloadBlob(response.blob, response.filename || `${modelName}_export.zip`);
    closeModal();
  } catch (error) {
    if (error instanceof ApiError && error.message.includes("metadata.sample_size")) {
      showSampleSizeModal((value) => exportTrainingResultArchive(resultId, modelName, value, run, showModal, closeModal), showModal, closeModal);
    } else {
      showModal({ title: "Ошибка экспорта", body: <p>{error instanceof Error ? error.message : "Неизвестная ошибка"}</p> });
    }
  }
}

function showSampleSizeModal(onSubmit: (sampleSize: number) => void, showModal: (modal: ModalState) => void, closeModal: () => void) {
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const sampleSize = Number.parseInt(String(data.get("sample_size") || ""), 10);
    if (!Number.isInteger(sampleSize) || sampleSize <= 0) {
      window.alert("sample_size должен быть положительным числом.");
      return;
    }
    closeModal();
    onSubmit(sampleSize);
  };
  showModal({
    title: "Нужен sample_size",
    body: (
      <form className="form-stack" onSubmit={submit}>
        <p>В checkpoint нет metadata.sample_size. Укажите размер входного тайла вручную.</p>
        <label className="field">
          <span>sample_size</span>
          <input name="sample_size" type="number" min="1" step="1" required />
        </label>
        <button className="primary" type="submit">
          Продолжить
        </button>
      </form>
    ),
  });
}
