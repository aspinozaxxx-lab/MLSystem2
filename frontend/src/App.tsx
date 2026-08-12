import {
  Activity,
  Archive,
  BarChart3,
  Check,
  ChevronDown,
  ChevronUp,
  Database,
  Download,
  ExternalLink,
  FileText,
  Layers3,
  ListChecks,
  LogOut,
  Play,
  Plus,
  RefreshCw,
  Save,
  Settings,
  Star,
  Trash2,
  X,
} from "lucide-react";
import {
  type FormEvent,
  type ReactNode,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ApiError, apiDownload, apiDownloadJson, apiForm, apiJson, downloadBlob } from "./api/client";
import type {
  AnyTemplate,
  AutomationRuleInfo,
  AutomationSnapshot,
  BootstrapInfo,
  DatasetResultsResponse,
  ConfigField,
  ConfigSchema,
  CustomDatasetInfo,
  DatasetCatalogInfo,
  DatasetEditorMutationResult,
  DatasetInfo,
  ImageryType,
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
  ResultClassInfo,
  ResultClassListResponse,
  TrainingResultBatchExportRequest,
  TrainingResultInfo,
  TrainingTemplate,
  TestSampleCatalogResponse,
  TestSampleBatchCreate,
  TestSampleBatchInfo,
  TestSampleBulkDownloadRequest,
  TestSampleDetail,
  TestSampleDownloadRequest,
  TestSampleDraftPreview,
  TestSampleEvaluationInfo,
  TestSampleMetric,
  TestSampleOptimizeRequest,
  TestSampleSummary,
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
  formatTrainingResultDate,
  integerOrNull,
  imageryTypeForInputChannels,
  isPrimaryDataset,
  isValidExportModelName,
  runningProgressLabel,
  shortVersion,
} from "./utils/format";
import {
  applyTestMarkupPreview,
  changeTestMarkupDownloadSelection,
  flattenTestMarkups,
  initialTestMarkupDownloadSelection,
  sortTestMarkupDatasets,
  testMarkupDownloadOptions,
  testMarkupDraft,
  testMarkupDraftChanged,
  testMarkupStats,
  type TestMarkupDownloadOption,
  type TestMarkupDraft,
} from "./utils/testMarkups";

const PROGRESS_REFRESH_MS = 10_000;
const TEST_SAMPLE_TILE_SIZES = [512, 768, 1024, 1536, 2048, 2560, 3072, 3584] as const;
const GROVIKA_LOGO_PATH = "/grovika/brand/grovika-lockup-horizontal-color.svg";

type ModalState = {
  title: string;
  body: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
};

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;

const DatasetEditorPage = lazy(() =>
  import("./DatasetEditorPage").then((module) => ({ default: module.DatasetEditorPage })),
);

type TestSampleBatchFormRow = {
  dataset: DatasetInfo;
  selected: boolean;
  minObjectCount: number;
};

function BrandLogo() {
  return <img className="brand-logo" src={GROVIKA_LOGO_PATH} alt="GROVIKA" width="190" height="60" />;
}

export function App() {
  const [route, setRoute] = useState(currentRoute());
  const [user, setUser] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [bootstrap, setBootstrap] = useState<BootstrapInfo | null>(null);
  const [modal, setModal] = useState<ModalState | null>(null);
  const routeGuardRef = useRef<(() => boolean) | null>(null);
  const acceptedHashRef = useRef(window.location.hash);

  const closeModal = useCallback(() => setModal(null), []);
  const registerRouteGuard = useCallback((guard: (() => boolean) | null) => {
    routeGuardRef.current = guard;
  }, []);

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
    const onHashChange = () => {
      const guard = routeGuardRef.current;
      if (guard && !guard()) {
        window.history.replaceState(null, "", acceptedHashRef.current || "#/");
        return;
      }
      acceptedHashRef.current = window.location.hash;
      setRoute(currentRoute());
    };
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
    if (routeGuardRef.current && !routeGuardRef.current()) return;
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
    return <LoadingPage text="Проверка сессии" branded />;
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
      registerRouteGuard={registerRouteGuard}
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
  registerRouteGuard: (guard: (() => boolean) | null) => void;
}) {
  const [head, second] = props.route;
  if (head === "start") return <StartPage {...props} />;
  if (head === "queue") return <QueuePage {...props} />;
  if (head === "templates") return <TemplatesPage {...props} />;
  if (head === "automation") return <AutomationPage {...props} />;
  if (head === "classes") return <ClassEditorPage {...props} />;
  if (head === "dataset-editor") {
    return (
      <Suspense fallback={<LoadingPage text="Загрузка редактора датасетов" />}>
        <DatasetEditorPage run={props.run} registerRouteGuard={props.registerRouteGuard} />
      </Suspense>
    );
  }
  if (head === "model-export") return <ModelExportPage {...props} />;
  if (head === "scene-list-export") return <SceneListExportPage {...props} />;
  if (head === "test-markups" && second === "create") {
    return <TestMarkupCreatePage {...props} />;
  }
  if (head === "test-markups" && second) {
    return <TestSampleEditorPage {...props} sampleId={second} />;
  }
  if (head === "test-markups") return <TestMarkupCatalogPage {...props} />;
  if (head === "results" && second) return <DatasetResultsPage {...props} datasetKey={decodeURIComponent(second)} />;
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
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const exportRouteActive =
    route[0] === "model-export" || route[0] === "scene-list-export" || route[0] === "test-markups";
  const navItems = [
    { href: "#/start", key: "start", label: "Запуск", icon: Play },
    { href: "#/queue", key: "queue", label: "Очередь", icon: ListChecks },
    { href: "#/templates", key: "templates", label: "Шаблоны", icon: Settings },
    { href: "#/automation", key: "automation", label: "Автоматизация", icon: Activity },
    { href: "#/classes", key: "classes", label: "Редактор классов", icon: Database },
    { href: "#/dataset-editor", key: "dataset-editor", label: "Редактор датасетов", icon: Layers3 },
  ];
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#/" aria-label="На главную">
          <BrandLogo />
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
          <div
            className={`nav-dropdown ${exportMenuOpen ? "open" : ""}`}
            onBlur={(event) => {
              if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
                setExportMenuOpen(false);
              }
            }}
          >
            <button
              className={exportRouteActive ? "active" : ""}
              type="button"
              aria-haspopup="menu"
              aria-expanded={exportMenuOpen}
              onClick={() => setExportMenuOpen((current) => !current)}
            >
              <Download size={16} />
              Экспорт
              <ChevronDown className="nav-dropdown-chevron" size={14} />
            </button>
            <div className="nav-dropdown-menu" role="menu">
              <a
                className={route[0] === "model-export" ? "active" : ""}
                href="#/model-export"
                role="menuitem"
                onClick={() => setExportMenuOpen(false)}
              >
                <Archive size={16} />
                Экспорт моделей
              </a>
              <a
                className={route[0] === "scene-list-export" ? "active" : ""}
                href="#/scene-list-export"
                role="menuitem"
                onClick={() => setExportMenuOpen(false)}
              >
                <FileText size={16} />
                Создать список сцен
              </a>
              <a
                className={route[0] === "test-markups" && route[1] === "create" ? "active" : ""}
                href="#/test-markups/create"
                role="menuitem"
                onClick={() => setExportMenuOpen(false)}
              >
                <Layers3 size={16} />
                Создание тестовых разметок
              </a>
              <a
                className={route[0] === "test-markups" && route[1] !== "create" ? "active" : ""}
                href="#/test-markups"
                role="menuitem"
                onClick={() => setExportMenuOpen(false)}
              >
                <Check size={16} />
                Тестовые разметки
              </a>
            </div>
          </div>
          <a className={route[0] === "results" ? "active" : ""} href="#/results">
            <BarChart3 size={16} />
            Результаты
          </a>
          <button type="button" title={`Выйти: ${user}`} onClick={onLogout}>
            <LogOut size={16} />
            Выйти
          </button>
        </nav>
      </header>
      <main className={`page ${route[0] === "dataset-editor" ? "page-wide" : ""}`}>{children}</main>
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
          <BrandLogo />
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

function LoadingPage({ text, branded = false }: { text: string; branded?: boolean }) {
  const status = (
    <div className="inline-row loading-status" role="status" aria-live="polite">
      <RefreshCw className="status-spinner" size={18} />
      <span>{text}</span>
    </div>
  );
  if (branded) {
    return (
      <main className="login-page loading-page">
        <section className="login-panel loading-panel">
          <div className="login-mark">
            <BrandLogo />
          </div>
          {status}
        </section>
      </main>
    );
  }
  return <section className="panel">{status}</section>;
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
        <a className="tool-card" href="/projects">
          <div>
            <div className="card-title">
              <Layers3 size={20} />
              Mapflow
            </div>
            <p className="muted">Открыть интерфейс ПМО и результаты обработок.</p>
          </div>
          <span className="secondary compact-action">
            <ExternalLink size={14} />
            Открыть
          </span>
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
  const selectedDataset = useMemo(
    () => bootstrap.datasets.find((item) => item.key === datasetKey),
    [bootstrap.datasets, datasetKey],
  );
  const trainingSchema = useMemo(
    () => trainingConfigSchema(template?.config_schema, selectedDataset?.task || "binary"),
    [selectedDataset?.task, template?.config_schema],
  );

  useEffect(() => {
    void run(() => apiJson<MLflowExperimentInfo[]>("/mlflow/experiments")).then((payload) => {
      const list = payload || [];
      setExperiments(list);
      setExperimentId(latestExperimentId(list));
    });
  }, [run]);

  useEffect(() => {
    const next = { ...(template?.default_config || {}) };
    if (selectedDataset?.task === "multiclass") {
      next["train.loss"] = "cross_entropy_dice";
    }
    setConfig(next);
  }, [datasetKey, selectedDataset?.task, template?.id]);

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
          {template && trainingSchema ? (
            <ConfigEditor schema={trainingSchema} value={config} onChange={setConfig} />
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

function SceneListExportPage({ run, showModal }: RoutedPageProps) {
  const [imageryType, setImageryType] = useState<ImageryType>("kanopus");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const geojson = formData.get("geojson");
    if (!(geojson instanceof File) || !geojson.name) {
      showModal({ title: "Ошибка", body: <p>Выберите GeoJSON с разметкой.</p> });
      return;
    }
    if (!geojson.name.toLocaleLowerCase("ru").endsWith(".geojson")) {
      showModal({ title: "Ошибка", body: <p>Файл разметки должен иметь расширение .geojson.</p> });
      return;
    }

    formData.set("include_footprints", "true");
    setBusy(true);
    setStatus("Поиск подходящих сцен...");
    try {
      const response = await run(() => apiDownload("/scene-list-export", formData));
      if (!response) {
        setStatus("");
        return;
      }
      const fallbackName = geojson.name.replace(/\.geojson$/i, ".zip");
      const filename = response.filename || fallbackName;
      downloadBlob(response.blob, filename);
      setStatus(`Скачан архив ${filename} с TXT и GeoJSON футпринтов`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Создать список сцен"
        subtitle="Найти подготовленные снимки и показать их покрытие в QGIS"
      />
      <form className="form-stack" onSubmit={submit}>
        <section className="panel">
          <PanelHeader
            title="Исходные данные"
            subtitle="ZIP содержит TXT с относительными путями и GeoJSON футпринтов снимков в WGS84"
          />
          <div className="form-grid">
            <label className="field">
              <span>Тип снимков</span>
              <select
                name="imagery_type"
                value={imageryType}
                disabled={busy}
                onChange={(event) => setImageryType(event.target.value as ImageryType)}
              >
                <option value="kanopus">Канопус</option>
                <option value="ortho">Ортофото</option>
              </select>
            </label>
            <label className="field">
              <span>GeoJSON с разметкой</span>
              <input
                name="geojson"
                type="file"
                accept=".geojson,application/geo+json"
                disabled={busy}
                required
              />
            </label>
          </div>
        </section>
        <div className="inline-row">
          <button className="primary" type="submit" disabled={busy}>
            <FileText size={16} />
            {busy ? "Создание списка..." : "Создать TXT и футпринты"}
          </button>
          {status ? <span className="info-box">{status}</span> : null}
        </div>
      </form>
    </>
  );
}

type ModelExportRow = {
  dataset: DatasetInfo;
  result: TrainingResultInfo | null;
  selected: boolean;
  modelName: string;
  sampleSize: string;
  context: string;
};

function ModelExportPage({ bootstrap, run, showModal }: RoutedPageProps) {
  const [rows, setRows] = useState<ModelExportRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  useEffect(() => {
    let cancelled = false;
    const datasets = bootstrap.classes.flatMap((item) => item.datasets || []);
    setLoading(true);
    void Promise.all(
      datasets.map(async (dataset): Promise<ModelExportRow> => {
        const payload = await run(() => apiJson<DatasetResultsResponse>(`/results/datasets/${encodeURIComponent(dataset.key)}`));
        const result = payload ? latestSuccessfulTrainingResult(payload.results) : null;
        return {
          dataset,
          result,
          selected: Boolean(result && isPrimaryDataset(dataset)),
          modelName: result ? defaultTrainingZipModelName(result, bootstrap.datasets) : "",
          sampleSize: result?.sample_size_hint ? String(result.sample_size_hint) : "",
          context: "",
        };
      }),
    )
      .then((items) => {
        if (!cancelled) setRows(items);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [bootstrap.classes, bootstrap.datasets, run]);

  const updateRow = (datasetKey: string, patch: Partial<ModelExportRow>) => {
    setRows((current) => current.map((row) => (row.dataset.key === datasetKey ? { ...row, ...patch } : row)));
  };

  const availableRows = rows.filter((row) => row.result);
  const selectedRows = rows.filter((row) => row.result && row.selected);
  const allAvailableSelected = availableRows.length > 0 && availableRows.every((row) => row.selected);

  const toggleAll = () => {
    const nextSelected = !allAvailableSelected;
    setRows((current) => current.map((row) => ({ ...row, selected: Boolean(row.result && nextSelected) })));
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedRows.length) {
      showModal({ title: "Ошибка", body: <p>Выберите хотя бы одну модель для экспорта.</p> });
      return;
    }
    const names = new Set<string>();
    const items: NonNullable<TrainingResultBatchExportRequest["items"]> = [];
    for (const row of selectedRows) {
      const modelName = row.modelName.trim();
      if (!isValidExportModelName(modelName)) {
        showModal({ title: "Ошибка", body: <p>Имя модели должно содержать только a-z, 0-9, дефис и подчеркивание.</p> });
        return;
      }
      if (names.has(modelName)) {
        showModal({ title: "Ошибка", body: <p>Имена моделей в общем архиве должны быть уникальными.</p> });
        return;
      }
      names.add(modelName);
      const sampleSize = parseExportSampleSize(row.sampleSize);
      if (sampleSize === undefined) {
        showModal({ title: "Ошибка", body: <p>sample_size должен быть положительным числом, кратным 32.</p> });
        return;
      }
      const context = parseExportContext(row.context);
      if (context === undefined) {
        showModal({ title: "Ошибка", body: <p>context должен быть целым неотрицательным числом.</p> });
        return;
      }
      items.push({
        result_id: row.result!.id,
        model_name: modelName,
        sample_size: sampleSize,
        context,
      });
    }

    setBusy(true);
    setStatus("Сборка архива...");
    try {
      const request: TrainingResultBatchExportRequest = { items };
      const response = await run(() => apiDownloadJson("/results/training/triton-zip", request));
      if (response) {
        downloadBlob(response.blob, response.filename || "models_export.zip");
        setStatus("Архив готов");
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="Экспорт моделей" subtitle="Основные сети классов" />
      <form className="form-stack" onSubmit={submit}>
        <section className="panel">
          <PanelHeader
            title="Модели"
            subtitle={loading ? "Загрузка результатов" : `Доступно к экспорту: ${availableRows.length}`}
            aside={
              <button className="secondary compact-action" type="button" disabled={!availableRows.length || busy} onClick={toggleAll}>
                {allAvailableSelected ? "Снять все" : "Выбрать все"}
              </button>
            }
          />
          {loading ? (
            <div className="empty-state">Загрузка моделей</div>
          ) : rows.length ? (
            <div className="table-wrap">
              <table className="model-export-table">
                <colgroup>
                  <col className="model-export-col-check" />
                  <col className="model-export-col-dataset" />
                  <col className="model-export-col-model" />
                  <col className="model-export-col-date" />
                  <col className="model-export-col-name" />
                  <col className="model-export-col-sample" />
                  <col className="model-export-col-sample" />
                </colgroup>
                <thead>
                  <tr>
                    <th aria-label="Выбрано"></th>
                    <th>Класс</th>
                    <th>Модель</th>
                    <th>Обучена</th>
                    <th>Имя выгрузки</th>
                    <th>sample_size</th>
                    <th>context</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr className={row.result ? "" : "disabled-row"} key={row.dataset.key}>
                      <td>
                        <input
                          type="checkbox"
                          checked={row.selected}
                          disabled={!row.result || busy}
                          aria-label={`Выбрать ${row.dataset.name}`}
                          onChange={(event) => updateRow(row.dataset.key, { selected: event.target.checked })}
                        />
                      </td>
                      <td>
                        <span className="source-lines">
                          <strong>{row.dataset.name}</strong>
                          {row.dataset.version ? <small className="muted technical-value">{shortVersion(row.dataset.version)}</small> : null}
                        </span>
                      </td>
                      <td>
                        {row.result ? (
                          <span className="source-lines">
                            <strong>{row.result.model_name}</strong>
                            <small className="muted">{row.result.architecture}</small>
                          </span>
                        ) : (
                          <span className="muted">Нет успешной модели</span>
                        )}
                      </td>
                      <td className="technical-value">{row.result ? formatDateTime(row.result.trained_at || row.result.created_at) : "—"}</td>
                      <td>
                        <input
                          value={row.modelName}
                          disabled={!row.result || busy}
                          pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
                          aria-label={`Имя выгрузки ${row.dataset.name}`}
                          onChange={(event) => updateRow(row.dataset.key, { modelName: event.target.value })}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="32"
                          step="32"
                          value={row.sampleSize}
                          disabled={!row.result || busy}
                          aria-label={`sample_size ${row.dataset.name}`}
                          onChange={(event) => updateRow(row.dataset.key, { sampleSize: event.target.value })}
                        />
                      </td>
                      <td>
                        <input
                          type="number"
                          min="0"
                          step="1"
                          value={row.context}
                          disabled={!row.result || busy}
                          placeholder="из checkpoint"
                          aria-label={`context ${row.dataset.name}`}
                          onChange={(event) => updateRow(row.dataset.key, { context: event.target.value })}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">Датасеты не найдены</div>
          )}
        </section>
        <div className="inline-row">
          <button className="primary" type="submit" disabled={busy || loading || !selectedRows.length}>
            <Archive size={16} />
            Собрать zip
          </button>
          {status ? <span className="info-box">{status}</span> : null}
        </div>
      </form>
    </>
  );
}

function TestMarkupCreatePage({ bootstrap, run }: RoutedPageProps) {
  const datasets = useMemo(
    () =>
      bootstrap.datasets
        .filter(
          (dataset) =>
            !dataset.is_custom &&
            Boolean(dataset.scenes_file) &&
            Boolean(dataset.annotation_file) &&
            !(dataset.diagnostics || []).length,
        )
        .sort((left, right) => testMarkupDatasetLabel(left).localeCompare(testMarkupDatasetLabel(right), "ru")),
    [bootstrap.datasets],
  );
  const [tileSize, setTileSize] = useState(1536);
  const [minImageCount, setMinImageCount] = useState(5);
  const [maxImageCount, setMaxImageCount] = useState(10);
  const [rows, setRows] = useState<TestSampleBatchFormRow[]>(() =>
    datasets.map((dataset) => ({ dataset, selected: false, minObjectCount: 150 })),
  );
  const [busy, setBusy] = useState(false);
  const [catalog, setCatalog] = useState<TestSampleCatalogResponse | null>(null);
  const [batch, setBatch] = useState<TestSampleBatchInfo | null>(null);

  const loadCatalog = useCallback(async () => {
    const payload = await run(() => apiJson<TestSampleCatalogResponse>("/test-samples"));
    if (payload) setCatalog(payload);
  }, [run]);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  useEffect(() => {
    setRows((current) =>
      datasets.map((dataset) => {
        const existing = current.find((item) => item.dataset.key === dataset.key);
        return existing || { dataset, selected: false, minObjectCount: 150 };
      }),
    );
  }, [datasets]);

  useEffect(() => {
    let cancelled = false;
    void apiJson<TestSampleBatchInfo>("/test-sample-batches/latest")
      .then((latest) => {
        if (cancelled) return;
        setBatch(latest);
        setTileSize(latest.tile_size);
        setMinImageCount(latest.min_image_count);
        setMaxImageCount(latest.image_count);
        setRows((current) =>
          current.map((row) => {
            const previous = (latest.items || []).find((item) => item.dataset_key === row.dataset.key);
            return previous
              ? { ...row, selected: false, minObjectCount: previous.min_object_count }
              : row;
          }),
        );
      })
      .catch((error: unknown) => {
        if (!(error instanceof ApiError) || error.status !== 404) {
          void run(() => Promise.reject(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [run]);

  const batchActive = batch?.status === "queued" || batch?.status === "running";
  useEffect(() => {
    if (!batchActive || !batch) return undefined;
    const timer = window.setTimeout(() => {
      void run(() => apiJson<TestSampleBatchInfo>(`/test-sample-batches/${batch.id}`)).then((updated) => {
        if (!updated) return;
        setBatch(updated);
        if (updated.status !== "queued" && updated.status !== "running") void loadCatalog();
      });
    }, 2_000);
    return () => window.clearTimeout(timer);
  }, [batch, batchActive, loadCatalog, run]);

  const updateRow = (datasetKey: string, update: Partial<(typeof rows)[number]>) => {
    setRows((current) => current.map((row) => (row.dataset.key === datasetKey ? { ...row, ...update } : row)));
  };
  const displayedRows = sortTestMarkupDatasets(rows.map((row) => row.dataset), catalog)
    .map((dataset) => rows.find((row) => row.dataset.key === dataset.key))
    .filter((row): row is TestSampleBatchFormRow => Boolean(row));

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    try {
      const request: TestSampleBatchCreate = {
        tile_size: tileSize as TestSampleBatchCreate["tile_size"],
        min_image_count: minImageCount,
        image_count: maxImageCount,
        items: rows
          .filter((row) => row.selected)
          .map((row) => ({
            dataset_key: row.dataset.key,
            min_object_count: row.minObjectCount,
            metric: row.dataset.quality_metric || "pixel",
          })),
      };
      const payload = await run(() =>
        apiJson<TestSampleBatchInfo>("/test-sample-batches", { method: "POST", body: request }),
      );
      if (payload) setBatch(payload);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader title="Создание тестовых разметок" subtitle="Групповая подготовка постоянных тестовых разметок" />
      <form className="form-stack" onSubmit={submit}>
        <section className="panel">
          <PanelHeader
            title="Групповое создание тестовых разметок"
            subtitle="Пул содержит до тройного максимума тайлов, итоговый состав выбирается в заданном диапазоне"
          />
          {datasets.length ? (
            <>
              <div className="form-grid test-sample-batch-settings">
                <label className="field">
                  <span>Размер квадратного тайла, пиксели</span>
                  <select value={tileSize} disabled={busy || batchActive} onChange={(event) => setTileSize(Number(event.target.value))}>
                    {TEST_SAMPLE_TILE_SIZES.map((size) => <option key={size} value={size}>{size} × {size}</option>)}
                  </select>
                </label>
                <label className="field">
                  <span>Минимум снимков в итоге</span>
                  <input type="number" min="1" max={maxImageCount} step="1" required value={minImageCount} disabled={busy || batchActive} onChange={(event) => setMinImageCount(Number(event.target.value))} />
                </label>
                <label className="field">
                  <span>Максимум снимков в итоге</span>
                  <input type="number" min={minImageCount} step="1" required value={maxImageCount} disabled={busy || batchActive} onChange={(event) => setMaxImageCount(Number(event.target.value))} />
                </label>
              </div>
              <p className="muted test-sample-batch-pool-hint">
                До оптимизации будет сформирован пул до {Math.max(0, maxImageCount * 3)} тайлов на датасет.
              </p>
              <div className="button-row test-sample-batch-select-actions">
                <button className="secondary compact-action" type="button" disabled={busy || batchActive} onClick={() => setRows((current) => current.map((row) => ({ ...row, selected: true })))}>Выбрать все</button>
                <button className="secondary compact-action" type="button" disabled={busy || batchActive} onClick={() => setRows((current) => current.map((row) => ({ ...row, selected: false })))}>Снять все</button>
              </div>
              <div className="test-sample-batch-grid">
                {displayedRows.map((row) => {
                  const stats = testMarkupStats(catalog, row.dataset.key);
                  return (
                  <div className={`test-sample-batch-row ${row.selected ? "" : "disabled-row"}`} key={row.dataset.key}>
                    <label className="test-sample-batch-choice">
                      <input type="checkbox" checked={row.selected} disabled={busy || batchActive} aria-label={`Создать разметку ${row.dataset.name}`} onChange={(event) => updateRow(row.dataset.key, { selected: event.target.checked })} />
                      <span className="source-lines">
                        <strong>{row.dataset.class_name || row.dataset.name}</strong>
                        <span>{row.dataset.dataset_name || row.dataset.name}</span>
                        <span className="test-markup-creation-status">
                          {stats.hasPrimary ? <><Star className="primary-star" size={13} fill="currentColor" />Основная есть</> : "Основной нет"}
                          <span>Разметок: {stats.count}</span>
                        </span>
                      </span>
                    </label>
                    <label className="test-sample-batch-field">
                      <span>Мин. объектов</span>
                      <input type="number" min="1" step="1" value={row.minObjectCount} disabled={busy || batchActive} aria-label={`Минимум объектов ${row.dataset.name}`} onChange={(event) => updateRow(row.dataset.key, { minObjectCount: Number(event.target.value) })} />
                    </label>
                    <label className="test-sample-batch-field">
                      <span>Основная метрика</span>
                      <input value={qualityMetricLabel(row.dataset.quality_metric)} readOnly disabled />
                    </label>
                  </div>
                  );
                })}
              </div>
            </>
          ) : <div className="empty-state">Нет датасетов с однозначными файлами сцен и положительной разметки.</div>}
        </section>
        <div className="button-row">
          <button className="primary" type="submit" disabled={busy || batchActive || minImageCount > maxImageCount || !rows.some((row) => row.selected)}>
            <Layers3 size={16} />
            {batchActive ? "Формирование..." : "Создать выбранные разметки"}
          </button>
        </div>
      </form>

      {batch ? <TestSampleBatchProgress batch={batch} /> : null}
    </>
  );
}

function TestMarkupCatalogPage({ run, showModal, closeModal }: RoutedPageProps) {
  const [catalog, setCatalog] = useState<TestSampleCatalogResponse | null>(null);
  const [downloadingAll, setDownloadingAll] = useState(false);

  const loadCatalog = useCallback(async (reconcile = false) => {
    const payload = await run(() => apiJson<TestSampleCatalogResponse>(
      reconcile ? "/test-samples/reconcile" : "/test-samples",
      reconcile ? { method: "POST" } : undefined,
    ));
    if (payload) setCatalog(payload);
  }, [run]);

  useEffect(() => {
    void loadCatalog(true);
  }, [loadCatalog]);

  const samples = flattenTestMarkups(catalog);
  const evaluationActive = samples.some(
    (sample) => sample.evaluation.status === "queued" || sample.evaluation.status === "running",
  );

  useEffect(() => {
    if (!evaluationActive) return undefined;
    const timer = window.setTimeout(() => void loadCatalog(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [evaluationActive, loadCatalog, catalog]);

  const downloadSelected = async (sampleIds: string[], includePreviews: boolean): Promise<boolean> => {
    const request: TestSampleBulkDownloadRequest = {
      sample_ids: sampleIds,
      include_previews: includePreviews,
    };
    setDownloadingAll(true);
    try {
      const payload = await run(() => apiDownloadJson("/test-samples/download", request));
      if (!payload) return false;
      downloadBlob(payload.blob, payload.filename || "тестовые_разметки.zip");
      return true;
    } finally {
      setDownloadingAll(false);
    }
  };

  const openBulkDownload = () => {
    showModal({
      title: "Скачать тестовые разметки",
      wide: true,
      body: (
        <BulkTestSampleDownloadForm
          catalog={catalog}
          onCancel={closeModal}
          onSubmit={downloadSelected}
        />
      ),
      footer: <></>,
    });
  };

  const removeSample = (sample: TestSampleSummary) => {
    showModal({
      title: "Удалить тестовую разметку",
      body: (
        <p>
          Разметка «{sample.name}» и все её файлы будут удалены без возможности восстановления.
          {sample.is_primary ? " Она назначена основной, поэтому тестовый F1 сетей станет недоступным до назначения новой." : ""}
        </p>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button
            className="danger"
            type="button"
            onClick={async () => {
              const deleted = await run(() => apiJson<null>(`/test-samples/${sample.id}`, { method: "DELETE" }));
              if (deleted !== undefined) {
                closeModal();
                await loadCatalog();
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
      <PageHeader title="Тестовые разметки" subtitle="Каталог постоянных разметок для независимой оценки сетей" />
      <section className="panel test-sample-catalog">
        <PanelHeader
          title="Каталог тестовых разметок"
          subtitle="Одна карточка соответствует одной сохранённой разметке"
          aside={
            <button
              className="secondary compact-action"
              type="button"
              disabled={downloadingAll || samples.length === 0}
              title={samples.length ? "Выбрать тестовые разметки для скачивания" : "Тестовые разметки ещё не созданы"}
              onClick={openBulkDownload}
            >
              <Download size={15} />
              {downloadingAll ? "Скачивание..." : "Скачать разметки"}
            </button>
          }
        />
        {catalog ? <TestSampleCatalog catalog={catalog} onDelete={removeSample} /> : <div className="empty-state">Загрузка каталога...</div>}
      </section>
    </>
  );
}

function TestSampleDownloadOptionsForm({
  onCancel,
  onSubmit,
}: {
  onCancel: () => void;
  onSubmit: (includePreviews: boolean) => Promise<boolean>;
}) {
  const [includePreviews, setIncludePreviews] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    try {
      if (await onSubmit(includePreviews)) onCancel();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form-stack" onSubmit={submit}>
      <DownloadModeFields
        includePreviews={includePreviews}
        onChange={setIncludePreviews}
      />
      <div className="button-row download-dialog-actions">
        <button className="secondary" type="button" disabled={submitting} onClick={onCancel}>Отмена</button>
        <button className="primary" type="submit" disabled={submitting}>
          <Download size={16} />
          {submitting ? "Формирование..." : "Скачать ZIP"}
        </button>
      </div>
    </form>
  );
}

function BulkTestSampleDownloadForm({
  catalog,
  onCancel,
  onSubmit,
}: {
  catalog: TestSampleCatalogResponse | null;
  onCancel: () => void;
  onSubmit: (sampleIds: string[], includePreviews: boolean) => Promise<boolean>;
}) {
  const options = useMemo(() => testMarkupDownloadOptions(catalog), [catalog]);
  const [selected, setSelected] = useState<Set<string>>(
    () => initialTestMarkupDownloadSelection(options),
  );
  const [includePreviews, setIncludePreviews] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const selectedIds = options
    .filter(({ sample }) => selected.has(sample.id))
    .map(({ sample }) => sample.id);

  const toggle = (sampleId: string, checked: boolean) => {
    setSelected((current) => changeTestMarkupDownloadSelection(
      options,
      current,
      { type: "toggle", sampleId, checked },
    ));
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedIds.length) return;
    setSubmitting(true);
    try {
      if (await onSubmit(selectedIds, includePreviews)) onCancel();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form-stack bulk-test-sample-download-form" onSubmit={submit}>
      <div className="download-selection-header">
        <span className="field-label">Разметки</span>
        <button
          className="secondary compact-action"
          type="button"
          disabled={!selectedIds.length || submitting}
          onClick={() => setSelected((current) => changeTestMarkupDownloadSelection(
            options,
            current,
            { type: "clear" },
          ))}
        >
          Снять все
        </button>
      </div>
      <div className="test-sample-download-grid">
        {options.map(({ datasetName, sample }: TestMarkupDownloadOption) => {
          const available = sample.enabled_image_count > 0;
          const displayName = `${sample.class_name}_${datasetName}`;
          const createdAt = formatDateTime(sample.created_at);
          return (
            <label
              className={`test-sample-download-choice ${available ? "" : "disabled-row"}`}
              key={sample.id}
              title={available ? sample.name : `${sample.name}: в разметке нет включённых тайлов`}
            >
              <input
                type="checkbox"
                checked={available && selected.has(sample.id)}
                disabled={!available || submitting}
                aria-label={`Выбрать разметку ${displayName}: ${sample.name}, создана ${createdAt}`}
                onChange={(event) => toggle(sample.id, event.target.checked)}
              />
              <span className="source-lines">
                <strong>
                  {displayName}
                  {sample.is_primary ? <Star className="primary-star" size={14} fill="currentColor" aria-label="Основная разметка" /> : null}
                </strong>
                <span className="test-sample-download-created-at">{createdAt}</span>
              </span>
            </label>
          );
        })}
      </div>
      <DownloadModeFields
        includePreviews={includePreviews}
        onChange={setIncludePreviews}
      />
      <div className="button-row download-dialog-actions">
        <button className="secondary" type="button" disabled={submitting} onClick={onCancel}>Отмена</button>
        <button className="primary" type="submit" disabled={submitting || !selectedIds.length}>
          <Download size={16} />
          {submitting ? "Формирование..." : `Скачать выбранные (${selectedIds.length})`}
        </button>
      </div>
    </form>
  );
}

function DownloadModeFields({
  includePreviews,
  onChange,
}: {
  includePreviews: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <fieldset className="download-mode-fieldset">
      <legend>Состав архива</legend>
      <label className="download-mode-choice">
        <input
          type="radio"
          name="download-mode"
          checked={includePreviews}
          onChange={() => onChange(true)}
        />
        <span><strong>С превью</strong><small>TIFF, GeoJSON, PNG-маска и JPEG-превью</small></span>
      </label>
      <label className="download-mode-choice">
        <input
          type="radio"
          name="download-mode"
          checked={!includePreviews}
          onChange={() => onChange(false)}
        />
        <span><strong>Без превью</strong><small>Только TIFF и GeoJSON</small></span>
      </label>
    </fieldset>
  );
}

function TestSampleBatchProgress({ batch }: { batch: TestSampleBatchInfo }) {
  const percent = batch.total_count ? Math.round((batch.completed_count / batch.total_count) * 100) : 0;
  return (
    <section className="panel test-sample-batch-progress">
      <PanelHeader
        title="Ход группового создания"
        subtitle={`${batch.completed_count} / ${batch.total_count} · итог ${batch.min_image_count}–${batch.image_count} тайлов · прошло ${formatElapsedSeconds(batch.elapsed_seconds)}`}
        aside={<span className={`badge ${batch.status === "ok" ? "ok" : batch.status === "error" ? "error" : batch.status === "partial" ? "warning" : "neutral"}`}>{batchStatusLabel(batch.status)}</span>}
      />
      <div className="batch-progress-track" aria-label={`Выполнено ${percent}%`}><span style={{ width: `${percent}%` }} /></div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Датасет</th><th>Статус</th><th>Пул</th><th>Результат</th></tr></thead>
          <tbody>
            {(batch.items || []).map((item) => (
              <tr key={item.id} title={item.error || undefined}>
                <td>{item.position}</td><td>{item.dataset_name}</td>
                <td><span className={`badge ${item.status === "ok" ? "ok" : item.status === "error" ? "error" : "neutral"}`}>{batchItemStatusLabel(item.status)}</span></td>
                <td>{item.pool_tile_count ?? "—"} тайлов · {item.pool_object_count ?? "—"} объектов</td>
                <td>{item.sample_id ? <a href={`#/test-markups/${item.sample_id}`}>{item.sample_name || "Открыть разметку"}</a> : item.error ? <span className="error-text">Ошибка — наведите для подробностей</span> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatElapsedSeconds(value: number): string {
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const seconds = value % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}` : `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function batchStatusLabel(status: TestSampleBatchInfo["status"]): string {
  return { queued: "в очереди", running: "выполняется", ok: "готово", partial: "частично", error: "ошибка" }[status];
}

function batchItemStatusLabel(status: NonNullable<TestSampleBatchInfo["items"]>[number]["status"]): string {
  return { queued: "в очереди", running: "выполняется", ok: "готово", error: "ошибка" }[status];
}

function TestSampleCatalog({
  catalog,
  onDelete,
}: {
  catalog: TestSampleCatalogResponse;
  onDelete: (sample: TestSampleSummary) => void;
}) {
  const samples = flattenTestMarkups(catalog).sort((left, right) => {
    const classOrder = left.class_name.localeCompare(right.class_name, "ru");
    if (classOrder) return classOrder;
    const datasetOrder = left.dataset_name.localeCompare(right.dataset_name, "ru");
    if (datasetOrder) return datasetOrder;
    return right.created_at.localeCompare(left.created_at);
  });
  if (!samples.length) return <div className="empty-state">Тестовые разметки ещё не созданы.</div>;
  return (
    <div className="test-markup-card-grid">
      {samples.map((sample) => (
        <article className="test-markup-card" key={sample.id}>
          <div className="test-markup-card-header">
            <a href={`#/test-markups/${sample.id}`}>
              <strong>{sample.dataset_name}</strong>
              <span>{sample.name}</span>
            </a>
            <div className="inline-row">
              {sample.is_primary ? <Star className="primary-star" size={20} fill="currentColor" aria-label="Основная разметка" /> : null}
              <button
                className="danger icon-button"
                type="button"
                aria-label={`Удалить разметку ${sample.name}`}
                title="Удалить разметку"
                onClick={() => onDelete(sample)}
              >
                <Trash2 size={15} />
              </button>
            </div>
          </div>
          <a className="test-markup-card-body" href={`#/test-markups/${sample.id}`}>
            <span><small>F1 pix</small><strong>{formatF1Score(sample.evaluation.pixel?.f1)}</strong></span>
            <span><small>F1 obj</small><strong>{formatF1Score(sample.evaluation.objects?.f1)}</strong></span>
            <span><small>Тайлы</small><strong>{sample.enabled_image_count}/{sample.image_count}</strong></span>
          </a>
          <div className="test-markup-card-footer">
            <TestSampleEvaluationBadge evaluation={sample.evaluation} />
            <span className="muted">
              {sample.evaluation.status !== "current" && (sample.evaluation.pixel || sample.evaluation.objects)
                ? "предыдущие значения · "
                : ""}
              {formatDateTime(sample.created_at)}
            </span>
          </div>
        </article>
      ))}
    </div>
  );
}

function TestSampleEditorPage({
  sampleId,
  run,
  showModal,
  closeModal,
  registerRouteGuard,
}: RoutedPageProps & { sampleId: string }) {
  const [sample, setSample] = useState<TestSampleDetail | null>(null);
  const [draft, setDraft] = useState<TestMarkupDraft | null>(null);
  const [draftEvaluation, setDraftEvaluation] = useState<TestSampleEvaluationInfo | null>(null);
  const [previewPending, setPreviewPending] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [recalculating, setRecalculating] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [minTileCount, setMinTileCount] = useState(1);
  const [maxTileCount, setMaxTileCount] = useState(1);
  const [minObjectCount, setMinObjectCount] = useState(1);

  const loadSample = useCallback(async () => {
    setLoaded(false);
    await run(() => apiJson<TestSampleCatalogResponse>(
      "/test-samples/reconcile",
      { method: "POST" },
    ));
    const payload = await run(() => apiJson<TestSampleDetail>(`/test-samples/${sampleId}`));
    if (payload) {
      setSample(payload);
      setDraft(testMarkupDraft(payload));
      setDraftEvaluation(null);
      setPreviewPending(false);
      if (payload.enabled_image_count > 0) {
        setMinTileCount(payload.enabled_image_count);
        setMaxTileCount(payload.enabled_image_count);
        setMinObjectCount(Math.max(1, payload.enabled_object_count));
      } else {
        setMinTileCount(1);
        setMaxTileCount(payload.image_count);
        setMinObjectCount(1);
      }
    }
    setLoaded(true);
  }, [run, sampleId]);

  useEffect(() => {
    void loadSample();
  }, [loadSample]);

  const refreshSample = useCallback(async () => {
    const payload = await run(() => apiJson<TestSampleDetail>(`/test-samples/${sampleId}`));
    if (payload) setSample(payload);
  }, [run, sampleId]);

  const evaluationActive = sample?.evaluation.status === "queued"
    || sample?.evaluation.status === "running";
  useEffect(() => {
    if (!evaluationActive) return undefined;
    const timer = window.setTimeout(() => void refreshSample(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [evaluationActive, refreshSample, sample]);

  const changed = Boolean(sample && draft && testMarkupDraftChanged(sample, draft));
  const dirty = changed || previewPending;

  useEffect(() => {
    if (!dirty) {
      registerRouteGuard(null);
      return undefined;
    }
    const confirmLeave = () => window.confirm("Есть несохранённые изменения тестовой разметки. Отбросить их?");
    registerRouteGuard(confirmLeave);
    return () => registerRouteGuard(null);
  }, [dirty, registerRouteGuard]);

  useEffect(() => {
    if (!dirty) return undefined;
    const beforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [dirty]);

  const invalidatePreview = () => {
    setDraftEvaluation(null);
    setPreviewPending(false);
  };

  const evaluate = async () => {
    if (!draft) return;
    setEvaluating(true);
    try {
      const payload = await run(() =>
        apiJson<TestSampleDraftPreview>(`/test-samples/${sampleId}/evaluate-preview`, {
          method: "POST",
          body: { enabled_tile_indices: draft.enabledTileIndices },
        }),
      );
      if (payload) {
        setDraftEvaluation(payload.evaluation);
        setPreviewPending(true);
      }
    } finally {
      setEvaluating(false);
    }
  };

  const recalculatePrimary = async () => {
    setRecalculating(true);
    try {
      const payload = await run(() =>
        apiJson<TestSampleDetail>(`/test-samples/${sampleId}/evaluate`, {
          method: "POST",
        }),
      );
      if (payload) setSample(payload);
    } finally {
      setRecalculating(false);
    }
  };

  const optimize = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setOptimizing(true);
    try {
      const request: TestSampleOptimizeRequest = {
        min_tile_count: minTileCount,
        max_tile_count: maxTileCount,
        min_object_count: minObjectCount,
        metric: sample?.quality_metric || "pixel",
      };
      const payload = await run(() =>
        apiJson<TestSampleDraftPreview>(`/test-samples/${sampleId}/optimize-preview`, {
          method: "POST",
          body: request,
        }),
      );
      if (payload) {
        setDraft((current) => current ? applyTestMarkupPreview(current, payload) : current);
        setDraftEvaluation(payload.evaluation);
        setPreviewPending(true);
      }
    } finally {
      setOptimizing(false);
    }
  };

  const download = async (includePreviews: boolean): Promise<boolean> => {
    if (!sample || !draft) return false;
    const request: TestSampleDownloadRequest = {
      enabled_tile_indices: draft.enabledTileIndices,
      include_previews: includePreviews,
    };
    setDownloading(true);
    try {
      const payload = await run(() => apiDownloadJson(sample.download_url, request));
      if (!payload) return false;
      downloadBlob(payload.blob, payload.filename || "test_markup.zip");
      return true;
    } finally {
      setDownloading(false);
    }
  };

  const openDownload = () => {
    showModal({
      title: "Скачать тестовую разметку",
      body: (
        <TestSampleDownloadOptionsForm
          onCancel={closeModal}
          onSubmit={download}
        />
      ),
      footer: <></>,
    });
  };

  const toggleTile = (tileIndex: number, enabled: boolean) => {
    setDraft((current) => {
      if (!current) return current;
      const selected = new Set(current.enabledTileIndices);
      if (enabled) selected.add(tileIndex);
      else selected.delete(tileIndex);
      return { ...current, enabledTileIndices: [...selected].sort((left, right) => left - right) };
    });
    invalidatePreview();
  };

  const rename = () => {
    if (!draft) return;
    showModal({
      title: "Переименовать тестовую разметку",
      body: (
        <RenameTestSampleForm
          initialName={draft.name}
          onCancel={closeModal}
          onSubmit={async (name) => {
            setDraft((current) => current ? { ...current, name } : current);
            closeModal();
          }}
        />
      ),
      footer: <></>,
    });
  };

  const changePrimary = () => {
    if (!draft) return;
    const makePrimary = !draft.isPrimary;
    showModal({
      title: makePrimary ? "Назначить основную разметку" : "Снять признак основной",
      body: (
        <p>
          {makePrimary
            ? "После общего сохранения эта разметка заменит текущую основную разметку датасета, а оценки сетей автоматически пересчитаются."
            : "После общего сохранения датасет останется без основной тестовой разметки, а его тестовые метрики станут недоступными."}
        </p>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button
            className={makePrimary ? "primary" : "danger"}
            type="button"
            onClick={() => {
              setDraft((current) => current ? { ...current, isPrimary: makePrimary } : current);
              closeModal();
            }}
          >
            Подтвердить
          </button>
        </>
      ),
    });
  };

  const save = async () => {
    if (!sample || !draft || !dirty) return;
    setSaving(true);
    try {
      const payload = await run(() =>
        apiJson<TestSampleDetail>(`/test-samples/${sample.id}`, {
          method: "PATCH",
          body: {
            name: draft.name.trim(),
            is_primary: draft.isPrimary,
            enabled_tile_indices: draft.enabledTileIndices,
          },
        }),
      );
      if (payload) {
        setSample(payload);
        setDraft(testMarkupDraft(payload));
        setDraftEvaluation(null);
        setPreviewPending(false);
      }
    } finally {
      setSaving(false);
    }
  };

  const remove = () => {
    if (!sample) return;
    showModal({
      title: "Удалить тестовую разметку",
      body: (
        <p>
          Разметка «{sample.name}» и все её файлы будут удалены без возможности восстановления.
          {dirty ? " Несохранённый черновик будет отброшен." : ""}
          {sample.is_primary ? " Это основная разметка: тестовый F1 всех сетей датасета станет недоступным." : ""}
        </p>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button
            className="danger"
            type="button"
            onClick={async () => {
              const deleted = await run(() => apiJson<null>(`/test-samples/${sample.id}`, { method: "DELETE" }));
              if (deleted !== undefined) {
                registerRouteGuard(null);
                closeModal();
                navigate("test-markups");
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

  if (!loaded) return <LoadingPage text="Загрузка тестовой разметки" />;
  if (!sample || !draft) {
    return (
      <>
        <PageHeader title="Тестовая разметка не найдена" />
        <a className="secondary" href="#/test-markups">Вернуться в каталог</a>
      </>
    );
  }

  const enabledIndices = new Set(draft.enabledTileIndices);
  const enabledTiles = (sample.tiles || []).filter((tile) => enabledIndices.has(tile.index));
  const hasEnabledTiles = enabledTiles.length > 0;
  const enabledObjectCount = enabledTiles.reduce((total, tile) => total + tile.object_count, 0);
  const savedEnabledIndices = testMarkupDraft(sample).enabledTileIndices;
  const compositionChanged = savedEnabledIndices.join(",") !== draft.enabledTileIndices.join(",");
  const optimizationValid =
    minTileCount > 0 &&
    maxTileCount >= minTileCount &&
    maxTileCount <= sample.image_count &&
    minObjectCount > 0;
  return (
    <>
      <PageHeader
        title={draft.name}
        subtitle={`${sample.dataset_name} · создана ${formatDateTime(sample.created_at)}`}
        actions={
          <>
            <a className="secondary" href="#/test-markups">Каталог</a>
            {dirty ? <span className="badge warning">Не сохранено</span> : null}
            <button className="primary" type="button" disabled={!dirty || saving || !draft.name.trim()} onClick={() => void save()}>
              <Save size={16} />
              {saving ? "Сохранение..." : "Сохранить"}
            </button>
            <button className={draft.isPrimary ? "danger" : "secondary"} type="button" disabled={saving} onClick={changePrimary}>
              {draft.isPrimary ? "Снять основную" : "Сделать основной"}
            </button>
            <button className="secondary" type="button" disabled={saving} onClick={rename}>Переименовать</button>
            <button className="danger" type="button" disabled={saving} onClick={remove}><Trash2 size={16} />Удалить</button>
          </>
        }
      />

      <section className="panel">
        <PanelHeader
          title="Состав разметки"
          subtitle={`${sample.dataset_name}${sample.dataset_version ? ` · версия ${sample.dataset_version}` : ""}`}
          aside={
            <div className="button-row">
              <button className="secondary" type="button" disabled={evaluating || saving || !hasEnabledTiles} onClick={() => void evaluate()}>
                <RefreshCw size={16} />
                {evaluating ? "Расчёт..." : "Оценить состав по псевдоразметке"}
              </button>
              <button className="primary" type="button" disabled={downloading || !hasEnabledTiles} onClick={openDownload}>
                <Download size={16} />
                {downloading ? "Скачивание..." : "Скачать ZIP"}
              </button>
            </div>
          }
        />
        <div className="metric-grid test-markup-summary">
          <Metric label="Назначение" value={draft.isPrimary ? <span className="badge ok">Основная</span> : "Обычная"} />
          <Metric label="Тайлы, включено / всего" value={`${enabledTiles.length} / ${sample.image_count}`} />
          <Metric label="Объекты, включено / всего" value={`${enabledObjectCount} / ${sample.actual_object_count}`} />
          <Metric label="Объекты, цель / факт" value={`${sample.requested_object_count} / ${sample.actual_object_count}`} />
          <Metric label="Размер тайла" value={`${sample.tile_width} × ${sample.tile_height}`} />
          <Metric label="Территории" value={sample.territory_count} />
        </div>
        {!hasEnabledTiles ? <div className="info-box">Включите хотя бы один тайл, чтобы рассчитать F1 и сохранить полезную разметку.</div> : null}
        {(sample.warnings || []).length ? (
          <div className="test-markup-warnings test-sample-warnings">
            {(sample.warnings || []).map((warning) => <div className="info-box" key={warning}>{warning}</div>)}
          </div>
        ) : null}
      </section>

      <section className="panel test-sample-optimizer">
        <PanelHeader
          title="Оптимизация состава"
          subtitle="Оптимизатор использует псевдоразметку основной сети, рассматривает все тайлы и подбирает состав с максимальным агрегированным F1"
        />
        <form className="form-stack" onSubmit={optimize}>
          <div className="form-grid">
            <label className="field">
              <span>Минимум тайлов</span>
              <input
                type="number"
                min="1"
                max={sample.image_count}
                step="1"
                required
                value={minTileCount}
                disabled={optimizing}
                onChange={(event) => setMinTileCount(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Максимум тайлов</span>
              <input
                type="number"
                min="1"
                max={sample.image_count}
                step="1"
                required
                value={maxTileCount}
                disabled={optimizing}
                onChange={(event) => setMaxTileCount(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Минимум объектов</span>
              <input
                type="number"
                min="1"
                step="1"
                required
                value={minObjectCount}
                disabled={optimizing}
                onChange={(event) => setMinObjectCount(Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>Основная метрика класса</span>
              <input value={qualityMetricLabel(sample.quality_metric)} readOnly disabled />
            </label>
          </div>
          <div className="button-row">
            <button className="primary" type="submit" disabled={optimizing || !optimizationValid}>
              <BarChart3 size={16} />
              {optimizing ? "Оптимизация..." : "Оптимизировать"}
            </button>
          </div>
        </form>
      </section>

      {draftEvaluation ? (
        <TestSampleEvaluationPanel evaluation={draftEvaluation} mode="pseudo" />
      ) : compositionChanged ? (
        <div className="info-box">Черновой состав ещё не оценён по псевдоразметке. Итоговые метрики ниже относятся к сохранённому составу.</div>
      ) : null}

      <TestSampleEvaluationPanel
        evaluation={sample.evaluation}
        mode="direct"
        recalculating={recalculating}
        recalculateDisabled={dirty || recalculating || evaluationActive || !hasEnabledTiles}
        onRecalculate={() => void recalculatePrimary()}
      />

      <section className="panel">
        <PanelHeader
          title="Тайлы"
          subtitle="Переключения изменяют только черновик до нажатия «Сохранить»"
        />
        <div className="markup-preview-grid">
          {(sample.tiles || []).map((tile) => (
            <article className={`markup-preview-card ${enabledIndices.has(tile.index) ? "" : "disabled"}`} key={tile.index}>
              <img src={tile.preview_url} alt={`Тайл ${tile.index}: ${tile.source_name}`} loading="lazy" />
              <div className="markup-preview-meta">
                <div className="test-sample-tile-heading">
                  <strong>Тайл {String(tile.index).padStart(3, "0")}</strong>
                  <label className="test-sample-toggle">
                    <input
                      type="checkbox"
                      checked={enabledIndices.has(tile.index)}
                      disabled={saving || optimizing || evaluating}
                      onChange={(event) => toggleTile(tile.index, event.target.checked)}
                    />
                    <span>{enabledIndices.has(tile.index) ? "Включён" : "Выключен"}</span>
                  </label>
                </div>
                <span title={tile.source_name}>{tile.source_name}</span>
                <small>{tile.territory}</small>
                <div className="test-sample-tile-badges">
                  <span className="badge neutral">Объектов: {tile.object_count}</span>
                  <span
                    className="badge neutral"
                    title={`${qualityMetricLabel(sample.quality_metric)} по псевдоразметке для оптимизации состава`}
                  >
                    {qualityMetricShort(sample.quality_metric)} псевдо: {formatF1Score(tile.f1_score)}
                  </span>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    </>
  );
}

function RenameTestSampleForm({
  initialName,
  onSubmit,
  onCancel,
}: {
  initialName: string;
  onSubmit: (name: string) => Promise<void>;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initialName);
  const [busy, setBusy] = useState(false);
  return (
    <form
      className="form-stack"
      onSubmit={async (event) => {
        event.preventDefault();
        setBusy(true);
        try {
          await onSubmit(name.trim());
        } finally {
          setBusy(false);
        }
      }}
    >
      <label className="field">
        <span>Название</span>
        <input autoFocus required maxLength={180} value={name} disabled={busy} onChange={(event) => setName(event.target.value)} />
      </label>
      <div className="button-row">
        <button className="secondary" type="button" disabled={busy} onClick={onCancel}>Отмена</button>
        <button className="primary" type="submit" disabled={busy || !name.trim()}>{busy ? "Применение..." : "Применить"}</button>
      </div>
    </form>
  );
}

function TestSampleEvaluationPanel({
  evaluation,
  mode,
  recalculating = false,
  recalculateDisabled = false,
  onRecalculate,
}: {
  evaluation: TestSampleEvaluationInfo;
  mode: "direct" | "pseudo";
  recalculating?: boolean;
  recalculateDisabled?: boolean;
  onRecalculate?: () => void;
}) {
  const direct = mode === "direct";
  const progress = evaluation.progress;
  return (
    <section className="panel test-sample-evaluation">
      <PanelHeader
        title={direct ? "Контрольные метрики" : "Предварительная оценка состава"}
        subtitle={direct
          ? "Итоговые метрики сохранённого состава рассчитываются прямым инференсом текущей основной сети класса"
          : "Оценка получена по существующей псевдоразметке основной сети и используется только для подбора состава"}
        aside={(
          <div className="button-row">
            <TestSampleEvaluationBadge evaluation={evaluation} />
            {direct && onRecalculate ? (
              <button
                className="secondary compact-action"
                type="button"
                disabled={recalculateDisabled}
                onClick={onRecalculate}
              >
                <RefreshCw size={15} />
                {recalculating ? "Постановка..." : "Пересчитать основной сетью"}
              </button>
            ) : null}
          </div>
        )}
      />
      <div className="test-sample-metric-grid">
        <TestSampleMetricCard title="Пиксельная метрика" metric={evaluation.pixel} />
        <TestSampleMetricCard
          title={`Объектная метрика · IoU ≥ ${evaluation.object_iou_threshold}`}
          metric={evaluation.objects}
        />
      </div>
      <div className="test-sample-evaluation-source">
        <span>
          <strong>{direct ? "Рассчитано сетью:" : "Псевдоразметка сети:"}</strong>{" "}
          {evaluation.model_name || (direct ? "ещё не рассчитано" : "нет подходящей псевдоразметки")}
        </span>
        {direct && evaluation.target_model_name && evaluation.target_model_name !== evaluation.model_name ? (
          <span><strong>Текущая основная сеть:</strong> {evaluation.target_model_name}</span>
        ) : null}
        {!direct && evaluation.markup_created_at ? <span><strong>Псевдоразметка:</strong> {formatDateTime(evaluation.markup_created_at)}</span> : null}
        {direct && evaluation.threshold != null ? <span><strong>Порог:</strong> {evaluation.threshold.toFixed(3)}</span> : null}
        {evaluation.evaluated_at ? <span><strong>Расчёт:</strong> {formatDateTime(evaluation.evaluated_at)}</span> : null}
      </div>
      {direct && progress?.total ? (
        <div className="info-box">
          Обработано тайлов: {progress.current ?? 0} / {progress.total}
          {progress.elapsed_minutes != null ? ` · прошло ${progress.elapsed_minutes} мин` : ""}
        </div>
      ) : null}
      {direct && evaluation.status !== "current" && (evaluation.pixel || evaluation.objects) ? (
        <div className="info-box">Показаны последние сохранённые значения; они относятся к предыдущей сети или ревизии состава.</div>
      ) : null}
      {evaluation.error ? <div className="info-box">{evaluation.error}</div> : null}
    </section>
  );
}

function TestSampleMetricCard({ title, metric }: { title: string; metric: TestSampleMetric | null | undefined }) {
  return (
    <div className="test-sample-metric-card">
      <h3>{title}</h3>
      <strong className="test-sample-f1">F1 {formatF1Score(metric?.f1)}</strong>
      <dl>
        <div><dt>Precision</dt><dd>{formatF1Score(metric?.precision)}</dd></div>
        <div><dt>Recall</dt><dd>{formatF1Score(metric?.recall)}</dd></div>
        <div><dt>TP</dt><dd>{metric?.true_positive ?? "—"}</dd></div>
        <div><dt>FP</dt><dd>{metric?.false_positive ?? "—"}</dd></div>
        <div><dt>FN</dt><dd>{metric?.false_negative ?? "—"}</dd></div>
      </dl>
    </div>
  );
}

function TestSampleEvaluationBadge({ evaluation }: { evaluation: TestSampleEvaluationInfo }) {
  const labels: Record<TestSampleEvaluationInfo["status"], string> = {
    current: "актуально",
    stale: "требует пересчёта",
    queued: "в очереди",
    running: "рассчитывается",
    unavailable: "оценка недоступна",
    error: "ошибка расчёта",
  };
  const classes: Record<TestSampleEvaluationInfo["status"], string> = {
    current: "ok",
    stale: "warning",
    queued: "neutral",
    running: "neutral",
    unavailable: "neutral",
    error: "error",
  };
  return <span className={`badge ${classes[evaluation.status]}`}>{labels[evaluation.status]}</span>;
}

function testMarkupDatasetLabel(dataset: DatasetInfo): string {
  return dataset.name;
}

function formatTestF1Percent(value: number): string {
  return (value * 100).toFixed(1);
}

function latestSuccessfulTrainingResult(results: TrainingResultInfo[]): TrainingResultInfo | null {
  const primary = results.find((item) => item.status === "ok" && item.is_primary);
  if (primary) return primary;
  return (
    [...results]
      .filter((item) => item.status === "ok")
      .sort((left, right) => trainingResultExportTime(right) - trainingResultExportTime(left))[0] || null
  );
}

function trainingResultExportTime(result: TrainingResultInfo): number {
  const timestamp = Date.parse(result.trained_at || result.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function qualityMetricLabel(metric: "pixel" | "objects" | null | undefined): string {
  return metric === "objects" ? "F1 объектовый" : "F1 пиксельный";
}

function qualityMetricShort(metric: "pixel" | "objects" | null | undefined): string {
  return metric === "objects" ? "F1 obj" : "F1 pix";
}

function parseExportSampleSize(value: string): number | null | undefined {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const sampleSize = Number.parseInt(trimmed, 10);
  if (!Number.isInteger(sampleSize) || sampleSize <= 0 || sampleSize % 32 !== 0) return undefined;
  return sampleSize;
}

function ClassEditorPage({ run, reloadBootstrap, showModal, closeModal }: RoutedPageProps) {
  const [catalog, setCatalog] = useState<DatasetCatalogInfo | null>(null);

  const loadCatalog = useCallback(async () => {
    const payload = await run(() => apiJson<DatasetCatalogInfo>("/dataset-catalog"));
    if (payload) setCatalog(payload);
  }, [run]);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  const applyCatalog = async (payload: DatasetCatalogInfo | undefined) => {
    if (!payload) return;
    setCatalog(payload);
    await reloadBootstrap();
  };

  const createClass = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const name = String(data.get("name") || "").trim();
    const imageryType = String(data.get("imagery_type") || "kanopus") as ImageryType;
    if (!name) return;
    const payload = await run(() =>
      apiJson<DatasetCatalogInfo>("/dataset-classes", {
        method: "POST",
        body: { name, imagery_type: imageryType },
      }),
    );
    if (payload) form.reset();
    await applyCatalog(payload);
  };

  const renameClass = async (event: FormEvent<HTMLFormElement>, classKey: string) => {
    event.preventDefault();
    const name = String(new FormData(event.currentTarget).get("name") || "").trim();
    await applyCatalog(await run(() =>
      apiJson<DatasetCatalogInfo>(`/dataset-classes/${encodeURIComponent(classKey)}`, {
        method: "PATCH",
        body: { name },
      }),
    ));
  };

  const updateClass = async (
    classKey: string,
    body: { quality_metric?: "pixel" | "objects"; imagery_type?: ImageryType },
  ) => {
    await applyCatalog(await run(() =>
      apiJson<DatasetCatalogInfo>(`/dataset-classes/${encodeURIComponent(classKey)}`, {
        method: "PATCH",
        body,
      }),
    ));
  };

  const changePrimary = async (classKey: string, datasetKey: string) => {
    if (!datasetKey) return;
    await applyCatalog(await run(() =>
      apiJson<DatasetCatalogInfo>(
        `/dataset-classes/${encodeURIComponent(classKey)}/primary-dataset`,
        { method: "PUT", body: { dataset_key: datasetKey } },
      ),
    ));
  };

  const openDatasetEditor = (classKey: string, dataset?: DatasetInfo) => {
    if (!catalog) return;
    const sources = (catalog.sources || []).filter(
      (source) => !source.assigned_dataset_key || source.assigned_dataset_key === dataset?.key,
    );
    const formId = `dataset-editor-${dataset?.key || classKey}`;
    showModal({
      title: dataset ? `Датасет «${dataset.name}»` : "Новый датасет",
      body: (
        <form
          id={formId}
          className="form-stack"
          onSubmit={async (event) => {
            event.preventDefault();
            const data = new FormData(event.currentTarget);
            const request = {
              name: String(data.get("name") || "").trim(),
              source_path: String(data.get("source_path") || ""),
            };
            const payload = await run(() =>
              apiJson<DatasetCatalogInfo>(
                dataset ? `/managed-datasets/${encodeURIComponent(dataset.key)}` : "/managed-datasets",
                {
                  method: dataset ? "PATCH" : "POST",
                  body: dataset ? request : { ...request, class_key: classKey },
                },
              ),
            );
            if (!payload) return;
            closeModal();
            await applyCatalog(payload);
          }}
        >
          <label>
            Название датасета
            <input
              name="name"
              defaultValue={dataset?.dataset_name || ""}
              maxLength={240}
              required
            />
          </label>
          <label>
            Источник MLMarkup
            <select name="source_path" defaultValue={dataset?.source_path || sources[0]?.key || ""} required>
              {sources.map((source) => (
                <option key={source.key} value={source.key}>
                  {source.name}
                  {(source.diagnostics || []).length ? " — требует внимания" : ""}
                </option>
              ))}
            </select>
          </label>
          {!sources.length ? <p className="error-text">Нет свободных папок MLMarkup.</p> : null}
          {(dataset?.diagnostics || []).length ? (
            <div className="notice warning">
              {(dataset?.diagnostics || []).map((diagnostic) => <div key={diagnostic}>{diagnostic}</div>)}
            </div>
          ) : null}
        </form>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button className="primary" type="submit" form={formId} disabled={!sources.length}>Сохранить</button>
        </>
      ),
    });
  };

  const confirmDatasetDeletion = (dataset: DatasetInfo) => {
    showModal({
      title: `Удалить датасет «${dataset.dataset_name || dataset.name}»?`,
      body: (
        <div className="form-stack">
          <p>
            Папка <strong>{dataset.source_path}</strong> будет удалена из MLMarkup отдельным Git-коммитом.
          </p>
          <div className="notice warning">
            Запись датасета, задания и результаты останутся в PostgreSQL и MLflow. Восстановления через
            интерфейс нет.
          </div>
          {dataset.is_primary ? (
            <p>После удаления у класса не будет основного датасета, пока вы не выберете другой.</p>
          ) : null}
        </div>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button
            className="danger"
            type="button"
            onClick={async () => {
              const result = await run(() =>
                apiJson<DatasetEditorMutationResult>(
                  `/dataset-editor/datasets/${encodeURIComponent(dataset.key)}`,
                  { method: "DELETE" },
                ),
              );
              if (!result) return;
              await loadCatalog();
              await reloadBootstrap();
              showModal({
                title: "Датасет удалён",
                body: (
                  <p>
                    Git-коммит <strong>{result.commit.slice(0, 8)}</strong> создан. Публикация MLMarkup:
                    {result.publication_status === "published" ? " завершена" : " выполняется"}.
                  </p>
                ),
              });
            }}
          >
            <Trash2 size={15} /> Удалить датасет
          </button>
        </>
      ),
    });
  };

  const synchronize = async () => {
    await applyCatalog(await run(() =>
      apiJson<DatasetCatalogInfo>("/dataset-catalog/sync", { method: "POST" }),
    ));
  };

  return (
    <>
      <PageHeader
        title="Редактор классов"
        subtitle="Классы, датасеты и источники снимков"
        actions={(
          <button className="secondary" type="button" onClick={synchronize}>
            <RefreshCw size={15} /> Синхронизировать MLMarkup
          </button>
        )}
      />
      <section className="panel">
        <PanelHeader title="Новый класс" subtitle="Выберите тип снимков, затем добавьте датасеты" />
        <form className="inline-form" onSubmit={createClass}>
          <input name="name" placeholder="Название класса" maxLength={240} required />
          <select name="imagery_type" defaultValue="kanopus" aria-label="Тип снимков">
            <option value="kanopus">Канопус</option>
            <option value="ortho">Ортофото</option>
          </select>
          <button className="primary" type="submit"><Plus size={15} /> Добавить класс</button>
        </form>
      </section>
      {!catalog ? <div className="empty-state">Загрузка каталога...</div> : null}
      {(catalog?.classes || []).map((classInfo) => {
        const datasets = classInfo.datasets || [];
        const hasFreeSource = (catalog?.sources || []).some((source) => !source.assigned_dataset_key);
        return (
          <section className="panel class-editor-card" key={classInfo.key}>
            <div className="class-editor-header">
              <form className="inline-form" onSubmit={(event) => renameClass(event, classInfo.key)}>
                <input name="name" defaultValue={classInfo.name} maxLength={240} required />
                <button className="secondary" type="submit">Переименовать</button>
              </form>
              <label>
                Тип снимков
                <select
                  value={classInfo.imagery_type || "kanopus"}
                  onChange={(event) => void updateClass(
                    classInfo.key,
                    { imagery_type: event.target.value as ImageryType },
                  )}
                >
                  <option value="kanopus">Канопус</option>
                  <option value="ortho">Ортофото</option>
                </select>
              </label>
              <label>
                Основная метрика
                <select
                  value={classInfo.quality_metric || "pixel"}
                  onChange={(event) => void updateClass(
                    classInfo.key,
                    { quality_metric: event.target.value as "pixel" | "objects" },
                  )}
                >
                  <option value="pixel">F1 пиксельный</option>
                  <option value="objects">F1 объектовый</option>
                </select>
              </label>
              <label>
                Основной датасет
                <select
                  value={classInfo.primary_dataset_key || ""}
                  disabled={!datasets.length}
                  onChange={(event) => void changePrimary(classInfo.key, event.target.value)}
                >
                  <option value="">Не назначен</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.key} value={dataset.key}>
                      {dataset.dataset_name || dataset.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="class-editor-datasets">
              {datasets.map((dataset) => (
                <article className="dataset-card" key={dataset.key}>
                  <div className="dataset-card-header">
                    <div className="source-lines">
                      <strong>{dataset.dataset_name || dataset.name}</strong>
                      <span className="muted">MLMarkup: {dataset.source_path}</span>
                      <span className="muted">Снимки: {imageryTypeLabel(classInfo.imagery_type)}</span>
                    </div>
                    <div className="inline-row">
                      {dataset.is_primary ? <span className="badge ok">основной</span> : null}
                      <span className={`badge ${(dataset.diagnostics || []).length ? "warning" : "ok"}`}>
                        {(dataset.diagnostics || []).length ? "требует внимания" : "источник доступен"}
                      </span>
                      <button
                        className="secondary"
                        type="button"
                        onClick={() => openDatasetEditor(classInfo.key, dataset)}
                      >
                        Параметры
                      </button>
                      <button
                        className="danger icon-button"
                        type="button"
                        aria-label={`Удалить датасет ${dataset.dataset_name || dataset.name}`}
                        title="Удалить папку датасета из MLMarkup"
                        onClick={() => confirmDatasetDeletion(dataset)}
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                  {(dataset.diagnostics || []).length ? (
                    <div className="notice warning">
                      {(dataset.diagnostics || []).map((diagnostic) => (
                        <div key={diagnostic}>{diagnostic}</div>
                      ))}
                    </div>
                  ) : null}
                </article>
              ))}
              {!datasets.length ? <div className="empty-state">Датасетов пока нет</div> : null}
            </div>
            <div className="button-row add-dataset-actions">
              <button
                className="secondary"
                type="button"
                disabled={!hasFreeSource}
                onClick={() => openDatasetEditor(classInfo.key)}
              >
                <Plus size={15} /> Добавить датасет
              </button>
            </div>
          </section>
        );
      })}
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
          <Metric label="Тип" value={job.purpose === "test_sample_f1" ? "тестовый F1" : job.purpose === "pseudo_markup" ? "разметка" : "обучение"} />
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

function ResultsPage({ run, showJobLog }: RoutedPageProps) {
  const [classes, setClasses] = useState<ResultClassInfo[] | null>(null);
  const [changes, setChanges] = useState<ResultChangeInfo[]>([]);
  useEffect(() => {
    void run(() => apiJson<ResultClassListResponse>("/results/classes")).then((payload) => {
      if (payload) setClasses(payload.classes || []);
    });
    void run(() => apiJson<ResultChangesResponse>("/results/changes")).then((payload) => {
      if (payload) setChanges(payload.changes || []);
    });
  }, [run]);

  return (
    <>
      <PageHeader title="Результаты" subtitle="Классы, датасеты и последние изменения" />
      <section className="content-grid">
        {(classes || []).map((item) => (
          <ResultClassCard item={item} key={item.key} />
        ))}
        {classes === null ? <div className="empty-state">Загрузка классов...</div> : null}
      </section>
      <section className="panel">
        <PanelHeader title="Последние изменения" />
        <ResultChangesTable changes={changes} showJobLog={showJobLog} />
      </section>
    </>
  );
}

function DatasetResultsPage({
  datasetKey,
  bootstrap,
  run,
  showModal,
  closeModal,
  showJobLog,
}: RoutedPageProps & { datasetKey: string }) {
  const [payload, setPayload] = useState<DatasetResultsResponse | null>(null);
  const load = useCallback(async () => {
    const data = await run(() => apiJson<DatasetResultsResponse>(`/results/datasets/${encodeURIComponent(datasetKey)}`));
    if (data) setPayload(data);
  }, [datasetKey, run]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!payload || !hasActiveDatasetResults(payload)) return undefined;
    const timer = window.setTimeout(() => void load(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [load, payload]);

  if (!payload) return <LoadingPage text="Загрузка результатов датасета" />;

  const showPseudo = (result: TrainingResultInfo) => {
    showModal({
      title: "Запустить pseudo-markup",
      body: (
        <PseudoMarkupForm
          datasetKey={datasetKey}
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

  const setPrimaryResult = async (result: TrainingResultInfo) => {
    if (result.is_primary) return;
    const updated = await run(() =>
      apiJson<TrainingResultInfo>(`/results/training/${result.id}/primary`, { method: "POST" }),
    );
    if (updated) await load();
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

  const recalculateTestF1 = async () => {
    const updated = await run(() =>
      apiJson<DatasetResultsResponse>(`/results/datasets/${encodeURIComponent(datasetKey)}/test-f1`, {
        method: "POST",
      }),
    );
    if (updated) setPayload(updated);
  };

  return (
    <>
      <PageHeader
        title={payload.dataset_name}
        subtitle={`Обновление датасета: ${formatDate(payload.dataset_updated_at)}`}
        actions={<a className="secondary" href="#/results">Все классы</a>}
      />
      {payload.primary_test_sample ? (
        <section className={`status-banner ${payload.test_f1_status === "current" ? "ok" : "error"}`}>
          <div>
            <strong>{payload.test_f1_status === "current" ? `${qualityMetricShort(payload.quality_metric)} (test) актуален` : payload.test_f1_status === "running" ? `Идёт пересчёт ${qualityMetricShort(payload.quality_metric)} (test)` : `${qualityMetricShort(payload.quality_metric)} (test) не актуален`}</strong>
            <span>Основная разметка: {payload.primary_test_sample.name} · {payload.primary_test_sample.enabled_image_count} тайлов</span>
          </div>
          {payload.test_f1_status !== "current" ? (
            <button className="primary" type="button" disabled={payload.test_f1_status === "running"} onClick={() => void recalculateTestF1()}>
              <RefreshCw size={16} />
              {payload.test_f1_status === "running" ? "Пересчёт..." : "Запустить пересчёт"}
            </button>
          ) : null}
        </section>
      ) : (
        <section className="status-banner neutral"><strong>Основная тестовая разметка не назначена</strong></section>
      )}
      <section className="panel">
        <ResultsTable
          payload={payload}
          datasets={bootstrap.datasets}
          imageFolders={bootstrap.image_folders}
          onPseudo={showPseudo}
          onZip={showZip}
          onPrimary={(result) => void setPrimaryResult(result)}
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
  registerRouteGuard: (guard: (() => boolean) | null) => void;
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
              <td className="technical-value">{job.queue_position}</td>
              <td>{statusBadge(job.status, job.type, job.progress)}</td>
              <td>{jobTypeBadge(job)}</td>
              <td>{queueDatasetCell(job)}</td>
              <td>{queueModelCell(job)}</td>
              <td className="technical-value">{formatDateTime(job.created_at)}</td>
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

function ResultClassCard({ item }: { item: ResultClassInfo }) {
  const datasets = item.datasets || [];
  return (
    <div className="class-card">
      <div className="card-title">
        <Layers3 size={20} />
        {item.name}
      </div>
      <p className="muted">Обновлено: {formatDate(item.updated_at)}</p>
      <div className="dataset-list">
        {datasets.length ? (
          datasets.map((dataset) => (
            <a className="dataset-link" href={`#/results/${encodeURIComponent(dataset.key)}`} key={dataset.key}>
              <span>{dataset.dataset_name || dataset.name}</span>
              {dataset.test_f1 !== null && dataset.test_f1 !== undefined ? (
                <strong className={`result-card-f1 ${dataset.test_f1_status === "current" ? "current" : "stale"}`}>{qualityMetricShort(dataset.quality_metric)} (test) {formatTestF1Percent(dataset.test_f1)}</strong>
              ) : null}
              <small>{integerOrNull(dataset.image_count) ?? "—"} снимков</small>
            </a>
          ))
        ) : (
          <div className="empty-state">Датасетов пока нет</div>
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
              onClick={() => navigate(`results/${encodeURIComponent(item.dataset_key)}`)}
            >
              <td onClick={(event) => event.stopPropagation()}>
                <span className="status-stack">
                  {resultStatusBadge(item.status, item.type, undefined, item.job_id, undefined, showJobLog)}
                  {sourceBadge(item.source)}
                </span>
              </td>
              <td>{item.class_name || item.class_key}</td>
              <td>{item.dataset_name}</td>
              <td>{item.model_name}</td>
              <td>{actionBadge(item.action, changeResultKind(item))}</td>
              <td className="technical-value">{formatDateTime(item.changed_at)}</td>
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
  onPrimary,
  onDeletePseudo,
  showJobLog,
}: {
  payload: DatasetResultsResponse;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  onPseudo: (result: TrainingResultInfo) => void;
  onZip: (result: TrainingResultInfo) => void;
  onPrimary: (result: TrainingResultInfo) => void;
  onDeletePseudo: (item: PseudoMarkupResultInfo) => void;
  showJobLog: (jobId: string) => Promise<void>;
}) {
  if (!payload.results.length) return <div className="empty-state">Для датасета пока нет результатов</div>;
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
                <col className="result-col-score" />
                <col className="result-col-epoch" />
                <col className="result-col-created" />
                <col className="result-col-actions" />
              </colgroup>
              <thead className="visually-hidden-header">
                <tr>
                  <th>МОДЕЛЬ</th>
                  <th>Статус</th>
                  <th>F1 (val)</th>
                  <th>F1 (test)</th>
                  <th>Epoch</th>
                  <th>Создано</th>
                  <th aria-label="Действия"></th>
                </tr>
              </thead>
              <tbody>
                <tr className="training-result-row">
                  <td title="МОДЕЛЬ">
                    <span className="source-lines">
                      <strong className="inline-row">
                        {result.status === "ok" ? (
                          <button
                            className="icon-button primary-result-star"
                            type="button"
                            title={result.is_primary ? "Основная сеть класса" : "Сделать основной сетью класса"}
                            aria-label={result.is_primary ? "Основная сеть класса" : "Сделать основной сетью класса"}
                            onClick={() => onPrimary(result)}
                          >
                            <Star className={result.is_primary ? "primary-star" : undefined} size={17} fill={result.is_primary ? "currentColor" : "none"} />
                          </button>
                        ) : null}
                        {result.model_name}
                      </strong>
                      <small className="muted">{result.architecture}</small>
                    </span>
                  </td>
                  <td title="Статус">
                    <span className="status-stack">
                      {resultStatusBadge(result.status, "training", result.progress, result.job_id, result.error, showJobLog)}
                      {sourceBadge(result.source)}
                    </span>
                  </td>
                  <td title={`${qualityMetricShort(result.quality_metric)} (val)`}>
                    <span className="source-lines"><small>{qualityMetricShort(result.quality_metric)} (val)</small><strong className="technical-value">{formatF1Score(result.f1_score)}</strong></span>
                  </td>
                  <td title={`${qualityMetricShort(result.quality_metric)} (test)`}>
                    {result.test_f1?.f1 !== null && result.test_f1?.f1 !== undefined ? (
                      <span className="source-lines">
                        <small>{qualityMetricShort(result.quality_metric)} (test)</small>
                        <span className={`badge technical-value ${result.test_f1.status === "current" ? "ok" : result.test_f1.status === "error" ? "error" : "warning"}`}>
                          {formatTestF1Percent(result.test_f1.f1)}
                        </span>
                      </span>
                    ) : result.test_f1?.status === "queued" || result.test_f1?.status === "running" ? (
                      <span className="badge neutral">расчёт</span>
                    ) : "—"}
                  </td>
                  <td className="technical-value" title="Epoch">{result.epoch ?? "—"}</td>
                  <td className="technical-value" title="Создано">{formatTrainingResultDate(result.status, result.trained_at, result.started_at, result.created_at)}</td>
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
                          {resultStatusBadge(item.status, "inference", item.progress, item.job_id, undefined, showJobLog)}
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
  datasetKey,
  result,
  datasets,
  imageFolders,
  run,
  closeModal,
  reload,
}: {
  datasetKey: string;
  result: TrainingResultInfo;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  run: Runner;
  closeModal: () => void;
  reload: () => Promise<void>;
}) {
  const imageryType = imageryTypeForInputChannels(result.input_channels);
  const compatibleDatasets = imageryType
    ? datasets.filter(
        (dataset) => dataset.key !== "custom" && dataset.imagery_type === imageryType,
      )
    : [];
  const compatibleFolders = imageryType
    ? imageFolders.filter((folder) => folder.imagery_type === imageryType)
    : [];
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const source = new FormData(event.currentTarget);
    const sourceDatasetKey = String(source.get("dataset_key") || "");
    const imageFolderKey = String(source.get("image_folder_key") || "");
    const file = source.get("scenes_txt");
    const hasFile = file instanceof File && Boolean(file.name);
    const sourceCount = [Boolean(sourceDatasetKey), Boolean(imageFolderKey), hasFile].filter(Boolean).length;
    if (sourceCount !== 1) {
      window.alert("Выберите ровно один источник: датасет, папку снимков или TXT.");
      return;
    }
    const request = new FormData();
    request.set("training_result_id", result.id);
    if (sourceDatasetKey) request.set("dataset_key", sourceDatasetKey);
    if (imageFolderKey) request.set("image_folder_key", imageFolderKey);
    if (hasFile && file instanceof File) request.set("scenes_txt", file);
    const created = await run(() => apiForm<JobDetail>(`/results/datasets/${encodeURIComponent(datasetKey)}/pseudo-markup`, request));
    if (created) {
      closeModal();
      await reload();
    }
  };
  return (
    <form className="form-stack" onSubmit={submit}>
      {imageryType ? (
        <p className="muted">
          Доступны только снимки типа «{imageryTypeLabel(imageryType)}», совместимые с {result.input_channels}-канальной моделью.
        </p>
      ) : (
        <p className="error-text">
          Для модели с {result.input_channels} входными каналами тип снимков не определён.
        </p>
      )}
      <label className="field">
        <span>Датасет</span>
        <select name="dataset_key" defaultValue="">
          <option value="">Не выбран</option>
          {compatibleDatasets.map((dataset) => (
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
          {compatibleFolders.map((folder) => (
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

function parseExportContext(value: string): number | null | undefined {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const context = Number(trimmed);
  if (!Number.isInteger(context) || context < 0) return undefined;
  return context;
}

export function trainingConfigSchema(
  schema: ConfigSchema | undefined,
  task: DatasetInfo["task"],
): ConfigSchema | undefined {
  if (!schema) return undefined;
  const allowedLosses =
    task === "multiclass"
      ? ["cross_entropy", "cross_entropy_dice"]
      : ["bce_dice", "focal_dice", "focal_tversky"];
  return {
    ...schema,
    fields: schema.fields.map((field) =>
      field.key === "train.loss"
        ? {
            ...field,
            options: allowedLosses,
            tooltip:
              task === "multiclass"
                ? "Multiclass loss: cross entropy, отдельно или вместе с Dice."
                : field.tooltip,
          }
        : field,
    ),
  };
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
  error: string | null | undefined,
  showJobLog: (jobId: string) => Promise<void>,
) {
  const badge = statusBadge(status, type, progress);
  if ((status === "error" || status === "failed") && jobId) {
    return (
      <button
        className="badge badge-button error"
        type="button"
        title={error || "Нажмите, чтобы открыть журнал ошибки"}
        onClick={() => void showJobLog(jobId)}
      >
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

function jobTypeBadge(job: JobSummary) {
  const label = job.purpose === "test_sample_f1" ? "тестовый F1" : job.purpose === "pseudo_markup" ? "разметка" : "обучение";
  return <span className={`badge ${job.type === "inference" ? "warning" : "neutral"}`}>{label}</span>;
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

function hasActiveDatasetResults(payload: DatasetResultsResponse): boolean {
  return payload.results.some(
    (item) =>
      isActiveStatus(item.status) ||
      isActiveStatus(item.test_f1?.status || "") ||
      (item.pseudo_markup_results || []).some((pseudo) => isActiveStatus(pseudo.status)),
  );
}

function datasetOptionLabel(item: DatasetInfo): string {
  const count = integerOrNull(item.image_count);
  return count === null ? item.name : `${item.name} (${count} img)`;
}

function imageFolderOptionLabel(item: ImageFolderInfo): string {
  return `${item.name} · ${imageryTypeLabel(item.imagery_type)} (${item.image_count} img)`;
}

function imageryTypeLabel(value: ImageryType | null | undefined): string {
  return value === "ortho" ? "Ортофото" : "Канопус";
}

function imageSourceLabel(item: PseudoMarkupResultInfo, datasets: DatasetInfo[], folders: ImageFolderInfo[]): string {
  const folder = folders.find((candidate) => candidate.key === item.source_dataset_name);
  const label = folder?.name
    || (item.dataset_key
      ? datasets.find((dataset) => dataset.key === item.dataset_key)?.name
      : undefined)
    || item.source_dataset_name;
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
  const suffix = dataset?.imagery_type === "ortho" ? "ortho" : "kanopus";
  if (!datasetPart || datasetPart === suffix || datasetPart.endsWith(`_${suffix}`)) return datasetPart || suffix;
  return `${datasetPart}_${suffix}`;
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
    const context = parseExportContext(String(data.get("context") || ""));
    if (context === undefined) {
      window.alert("context должен быть целым неотрицательным числом.");
      return;
    }
    await exportTrainingResultArchive(result.id, modelName, null, context, run, showModal, closeModal);
  };
  showModal({
    title: "Собрать Triton zip",
    body: (
      <form className="form-stack" onSubmit={submit}>
        <label className="field">
          <span>Имя модели</span>
          <input name="model_name" defaultValue={defaultName} pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?" required />
        </label>
        <label className="field">
          <span>context (необязательно)</span>
          <input name="context" type="number" min="0" step="1" placeholder="из checkpoint; для старого по умолчанию 0" />
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
  context: number | null,
  run: Runner,
  showModal: (modal: ModalState) => void,
  closeModal: () => void,
) {
  const request = new FormData();
  request.set("model_name", modelName);
  if (sampleSize !== null) request.set("sample_size", String(sampleSize));
  if (context !== null) request.set("context", String(context));
  try {
    const response = await apiDownload(`/results/training/${encodeURIComponent(resultId)}/triton-zip`, request);
    downloadBlob(response.blob, response.filename || `${modelName}_export.zip`);
    closeModal();
  } catch (error) {
    if (error instanceof ApiError && error.message.includes("metadata.sample_size")) {
      showSampleSizeModal((value) => exportTrainingResultArchive(resultId, modelName, value, context, run, showModal, closeModal), showModal, closeModal);
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
