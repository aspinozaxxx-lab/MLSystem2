import {
  Activity,
  Archive,
  BarChart3,
  Check,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileText,
  Images,
  Layers3,
  ListChecks,
  LoaderCircle,
  LogOut,
  Map as MapIcon,
  PencilLine,
  Play,
  Plus,
  RefreshCw,
  Save,
  Settings,
  Square,
  Star,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
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
  AutomationRuleInfo,
  AutomationSnapshot,
  BootstrapInfo,
  DatasetResultsResponse,
  CustomDatasetInfo,
  DatasetCatalogInfo,
  DatasetEditorCopyResult,
  DatasetEditorMutationResult,
  DatasetEditorUserDraftInfo,
  DatasetEditorUserDraftListResponse,
  DatasetInfo,
  ImageryType,
  ImageFolderInfo,
  JobDetail,
  JobLogInfo,
  JobSummary,
  JsonRecord,
  ModelInfo,
  PseudoMarkupResultInfo,
  QueueCountInfo,
  QueueSnapshot,
  ResultChangeInfo,
  ResultChangesResponse,
  ResultClassInfo,
  ResultClassListResponse,
  TrainingResultBatchExportRequest,
  TrainingResultInfo,
  TrainingContinuationOptions,
  TrainingTemplate,
  TestSampleCard,
  TestSampleDetail,
  TestSampleDownloadRequest,
  TestSampleDraftPreview,
  TestSampleEvaluationInfo,
  TestSampleMetric,
  TestSampleOptimizeRequest,
} from "./api/types";
import { TrainingContinuationForm } from "./TrainingContinuationForm";
import { PseudoComparisonProvider, PseudoComparisonTray, PseudoCompareButton } from "./PseudoComparisonSelection";
import { trainingResultFamilies } from "./utils/trainingResults";
import {
  defaultTrainingZipModelName,
  displayStoredFileName,
  formatDate,
  formatDateTime,
  formatF1Score,
  formatTestF1Percent,
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
  containedImageOneToOneScale,
  testMarkupDraft,
  testMarkupDraftChanged,
  type TestMarkupDraft,
} from "./utils/testMarkups";

import { ConfigEditor } from "./ConfigEditor";
import { TrainingLaunchForm } from "./TrainingLaunchForm";
import { InferenceTemplates } from "./InferenceTemplates";
import { inferenceTemplateForDataset } from "./utils/inferenceTemplates";
import { TestMarkupCreatePage } from "./TestMarkupCreatePage";
import { useTestMarkupClasses } from "./useTestMarkupClasses";
import { NewsPage, NewsSection } from "./News";
import { FeedbackButton, FeedbackSection } from "./Feedback";
import { useUsage } from "./useUsage";
import { usagePageViewed } from "./utils/usage";
import { configFieldTooltip, trainingConfigForTemplate, trainingConfigSchema } from "./utils/trainingConfig";

const PROGRESS_REFRESH_MS = 10_000;
const GROVIKA_LOGO_PATH = "/grovika/brand/grovika-lockup-horizontal-color.svg";

type ModalState = {
  title: string;
  body: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
  fullscreen?: boolean;
};

type Runner = <T>(operation: () => Promise<T>) => Promise<T | undefined>;

const DatasetEditorPage = lazy(() =>
  import("./DatasetEditorPage").then((module) => ({ default: module.DatasetEditorPage })),
);
const PseudoMarkupPage = lazy(() => import("./PseudoMarkupPage").then((module) => ({ default: module.PseudoMarkupPage })));
const PseudoComparisonPage = lazy(() => import("./PseudoComparisonPage").then((module) => ({ default: module.PseudoComparisonPage })));
const TestF1Page = lazy(() => import("./TestF1Page").then((module) => ({ default: module.TestF1Page })));

function BrandLogo() {
  return <img className="brand-logo" src={GROVIKA_LOGO_PATH} alt="GROVIKA" width="190" height="60" />;
}

export function App() {
  const [route, setRoute] = useState(currentRoute());
  const [user, setUser] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [bootstrap, setBootstrap] = useState<BootstrapInfo | null>(null);
  const bootstrapRequestRef = useRef<Promise<BootstrapInfo | undefined> | null>(null);
  const sessionUserRef = useRef(user);
  sessionUserRef.current = user;
  const [modal, setModal] = useState<ModalState | null>(null);
  const routeGuardRef = useRef<(() => boolean) | null>(null);
  const acceptedHashRef = useRef(window.location.hash);
  useUsage(user, route);

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

  const fetchBootstrap = useCallback((): Promise<BootstrapInfo | undefined> => {
    if (bootstrapRequestRef.current) return bootstrapRequestRef.current;
    const requestUser = sessionUserRef.current;
    const request = run(() => apiJson<BootstrapInfo>("/bootstrap")).then((payload) => {
      if (requestUser !== sessionUserRef.current) return undefined;
      if (payload) setBootstrap(payload);
      return payload;
    }).finally(() => {
      if (bootstrapRequestRef.current === request) bootstrapRequestRef.current = null;
    });
    bootstrapRequestRef.current = request;
    return request;
  }, [run]);
  const loadBootstrap = useCallback(async () => { await fetchBootstrap(); }, [fetchBootstrap]);
  const getBootstrap = useCallback(() => bootstrap ? Promise.resolve(bootstrap) : fetchBootstrap(), [bootstrap, fetchBootstrap]);
  const independentCatalogPage = route[0] === "test-markups" || route[0] === "results";

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
    if (user && !bootstrap && !independentCatalogPage) {
      void loadBootstrap();
    }
  }, [bootstrap, loadBootstrap, user, independentCatalogPage]);

  const logout = async () => {
    if (routeGuardRef.current && !routeGuardRef.current()) return;
    await run(() => apiJson<{ status: string }>("/auth/logout", { method: "POST" }));
    setUser(null);
    sessionUserRef.current = null;
    bootstrapRequestRef.current = null;
    setBootstrap(null);
    void import("./utils/rasterCache").then(({ rasterCache }) => rasterCache.clear());
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

  const page = (
    <RoutedPage
      route={route}
      username={user}
      bootstrap={bootstrap}
      run={run}
      reloadBootstrap={loadBootstrap}
      getBootstrap={getBootstrap}
      showModal={setModal}
      closeModal={closeModal}
      showJobLog={showJobLog}
      registerRouteGuard={registerRouteGuard}
    />
  );

  return (
    <PseudoComparisonProvider key={user}><Shell user={user} route={route} onLogout={logout} run={run}>
      {page}
      <Modal modal={modal} onClose={closeModal} />
    </Shell></PseudoComparisonProvider>
  );
}

function RoutedPage(context: {
  route: string[];
  username: string;
  bootstrap: BootstrapInfo | null;
  run: Runner;
  reloadBootstrap: () => Promise<void>;
  getBootstrap: () => Promise<BootstrapInfo | undefined>;
  showModal: (modal: ModalState) => void;
  closeModal: () => void;
  showJobLog: (jobId: string) => Promise<void>;
  registerRouteGuard: (guard: (() => boolean) | null) => void;
}) {
  const [head, second] = context.route;
  if (head === "test-markups" && second === "create") return <TestMarkupCreatePage run={context.run} />;
  if (head === "test-markups" && !second) return <TestMarkupCatalogPage {...context} />;
  if (head === "test-markups" && second) return <TestSampleEditorPage {...context} sampleId={second} />;
  if (head === "results" && second) return <DatasetResultsPage {...context} datasetKey={decodeURIComponent(second)} />;
  if (head === "results") return <ResultsPage {...context} />;
  if (!context.bootstrap) return <LoadingPage text="Загрузка справочников" />;
  const props = { ...context, bootstrap: context.bootstrap };
  if (head === "news") return <NewsPage slug={second} />;
  if (head === "feedback") return <FeedbackSection feedbackId={second} />;
  if (head === "pseudo-markup" && second === "compare") return <Suspense fallback={<LoadingPage text="Загрузка сравнения псевдоразметок" />}><PseudoComparisonPage ids={props.route[2] ?? ""} username={props.username} /></Suspense>;
  if (head === "pseudo-markup" && second) return <Suspense fallback={<LoadingPage text="Загрузка просмотра псевдоразметки" />}><PseudoMarkupPage resultId={second} username={props.username} /></Suspense>;
  if (head === "test-f1" && second) return <Suspense fallback={<LoadingPage text="Загрузка просмотра тестового F1" />}><TestF1Page resultId={second} username={props.username} /></Suspense>;
  if (head === "start") return <StartPage {...props} />;
  if (head === "queue") return <QueuePage {...props} />;
  if (head === "templates") return <TemplatesPage {...props} />;
  if (head === "automation") return <AutomationPage {...props} />;
  if (head === "classes") return <ClassEditorPage {...props} />;
  if (head === "dataset-editor") {
    return (
      <Suspense fallback={<LoadingPage text="Загрузка редактора датасетов" />}>
        <DatasetEditorPage
          run={props.run}
          registerRouteGuard={props.registerRouteGuard}
          initialDatasetKey={second ? decodeURIComponent(second) : undefined}
        />
      </Suspense>
    );
  }
  if (head === "model-export") return <ModelExportPage {...props} />;
  if (head === "scene-list-export") return <SceneListExportPage {...props} />;
  if (head === "jobs" && second) return <JobPage {...props} jobId={second} />;
  return <HomePage {...props} />;
}

function Shell({
  user,
  route,
  onLogout,
  run,
  children,
}: {
  user: string;
  route: string[];
  onLogout: () => void;
  run: Runner;
  children: ReactNode;
}) {
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const [queueCount, setQueueCount] = useState(0);
  const loadQueueCount = useCallback(async () => {
    const payload = await run(() => apiJson<QueueCountInfo>("/queues/count"));
    if (payload) setQueueCount(payload.active_jobs);
  }, [run]);

  useEffect(() => {
    void loadQueueCount();
    const timer = window.setInterval(() => void loadQueueCount(), PROGRESS_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [loadQueueCount, route]);
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
          <img className="brand-mark" src="/grovika/favicon/01-stepped-g.svg" alt="" width="28" height="28" />
        </a>
        <nav className="nav" aria-label="Основное меню">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <a className={route[0] === item.key ? "active" : ""} href={item.href} key={item.key} title={item.label} aria-label={item.label} aria-current={route[0] === item.key ? "page" : undefined}>
                <Icon size={16} />
                <span className="nav-label">{item.label}</span>
                {item.key === "queue" ? (
                  <span className="nav-queue-count" aria-label={`Активных заданий: ${queueCount}`}>
                    {queueCount}
                  </span>
                ) : null}
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
              aria-label="Экспорт"
              title="Экспорт"
              aria-haspopup="menu"
              aria-expanded={exportMenuOpen}
              onClick={() => setExportMenuOpen((current) => !current)}
            >
              <Download size={16} />
              <span className="nav-label">Экспорт</span>
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
          <a className={route[0] === "results" ? "active" : ""} href="#/results" aria-label="Результаты" title="Результаты" aria-current={route[0] === "results" ? "page" : undefined}>
            <BarChart3 size={16} />
            <span className="nav-label">Результаты</span>
          </a>
          <FeedbackButton />
          <button type="button" title={`Выйти: ${user}`} aria-label={`Выйти: ${user}`} onClick={onLogout}>
            <LogOut size={16} />
            <span className="nav-label">Выйти</span>
          </button>
        </nav>
      </header>
      <main className={`page ${["dataset-editor", "pseudo-markup", "test-f1"].includes(route[0]) || (route[0] === "test-markups" && !route[1]) ? "page-wide" : route[0] === "start" ? "training-page" : ""}`}>{route[0] === "results" ? <PseudoComparisonTray /> : null}{children}</main>
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

function HomePage({ bootstrap }: RoutedPageProps) {
  const links = useMemo(() => Object.fromEntries(bootstrap.links.map((item) => [item.key, item])), [bootstrap.links]);

  return (
    <>
      <div className="home-heading">
        <PageHeader title="Рабочая панель" subtitle="Обучение, очереди и результаты MLSystem2" actions={<>
          <a className="secondary home-service-link" href={links.grafana?.url} target="_blank" rel="noreferrer"
            aria-label="Grafana" title="Grafana — мониторинг сервера" aria-disabled={!links.grafana?.url}>
            <BarChart3 size={20} />
            <span className="home-service-label">Grafana</span>
          </a>
          <a className="secondary home-service-link" href={links.images?.url} target="_blank" rel="noreferrer"
            aria-label="Снимки" title="Снимки — файлы на сервере" aria-disabled={!links.images?.url}>
            <Images size={20} />
            <span className="home-service-label">Снимки</span>
          </a>
        </>} />
      </div>
      <FeedbackSection />
      <NewsSection />
    </>
  );
}

function StartPage({ bootstrap, run, reloadBootstrap, showModal, closeModal }: RoutedPageProps) {
  const [architecture, setArchitecture] = useState(bootstrap.models[0]?.architecture || "");
  const [datasetKey, setDatasetKey] = useState(bootstrap.datasets[0]?.key || "");
  const [config, setConfig] = useState<JsonRecord>({});
  const [busy, setBusy] = useState(false);
  const [runInferenceAfterTraining, setRunInferenceAfterTraining] = useState(false);
  const [secondaryPriority, setSecondaryPriority] = useState(false);
  const pipelineChoice = useRef<string | null>(null);
  const pretrainedChoice = useRef<boolean | null>(null);

  const template = useMemo(
    () => templateFor(bootstrap.training_templates, architecture, datasetKey),
    [architecture, bootstrap.training_templates, datasetKey],
  );
  const selectedDataset = useMemo(
    () => bootstrap.datasets.find((item) => item.key === datasetKey),
    [bootstrap.datasets, datasetKey],
  );
  const inferenceAvailable = Boolean(inferenceTemplateForDataset(bootstrap.inference_templates, bootstrap.datasets, datasetKey));
  useEffect(() => { if (!inferenceAvailable) setRunInferenceAfterTraining(false); }, [inferenceAvailable]);
  const trainingSchema = useMemo(
    () => trainingConfigSchema(template?.config_schema, selectedDataset?.task || "binary", String(config["train.pipeline_variant"] || "legacy")),
    [selectedDataset?.task, template?.config_schema, config["train.pipeline_variant"]],
  );

  useEffect(() => {
    setConfig(trainingConfigForTemplate(template, selectedDataset?.task || "binary", pipelineChoice.current, pretrainedChoice.current));
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
        showModal({ title: "Ошибка", body: <p>Для своего датасета нужны GeoJSON и TXT со снимками.</p> });
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

    const created = await run(() =>
      apiJson<JobDetail>("/training-jobs", {
        method: "POST",
        body: {
          mlflow_experiment_name: "MLSystem2",
          dataset_key: datasetKey,
          custom_dataset_id: customDatasetId,
          architecture,
          config,
          run_inference_after_training: runInferenceAfterTraining && inferenceAvailable,
          secondary_priority: secondaryPriority,
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
            Открыть задание
          </a>
        </>
      ),
    });
    await reloadBootstrap();
  };

  return (
    <>
      <PageHeader title="Запуск обучения" subtitle="Настройте модель, подготовку данных и условия завершения" />
      <TrainingLaunchForm
        models={bootstrap.models}
        datasets={bootstrap.datasets}
        architecture={architecture}
        onArchitectureChange={setArchitecture}
        datasetKey={datasetKey}
        onDatasetChange={setDatasetKey}
        template={template}
        schema={trainingSchema}
        value={config}
        onChange={(next) => {
          pipelineChoice.current = String(next["train.pipeline_variant"] || "legacy");
          if (next["train.pretrained"] !== config["train.pretrained"]) {
            pretrainedChoice.current = Boolean(next["train.pretrained"]);
          }
          setConfig(next);
        }}
        inferenceAvailable={inferenceAvailable}
        runInferenceAfterTraining={runInferenceAfterTraining && inferenceAvailable}
        onRunInferenceChange={setRunInferenceAfterTraining}
        secondaryPriority={secondaryPriority}
        onSecondaryPriorityChange={setSecondaryPriority}
        busy={busy}
        onSubmit={submit}
      />
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
                            <strong>{trainingModelLabel(row.result.model_name, row.result.pipeline_variant)}</strong>
                            <small className="muted">{row.result.architecture}</small>
                          </span>
                        ) : (
                          <span className="muted">Нет успешной модели</span>
                        )}
                      </td>
                      <td className="technical-value" data-label="Обучена">{row.result ? formatDateTime(row.result.trained_at || row.result.created_at) : "—"}</td>
                      <td data-label="Имя выгрузки">
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

function TestMarkupCatalogPage({ run, showModal, closeModal }: Pick<RoutedPageProps, "run" | "showModal" | "closeModal">) {
  const { index, classKey, setClassKey, loadClasses } = useTestMarkupClasses(run);
  const [cards, setCards] = useState<Record<string, TestSampleCard[]>>({});
  const requests = useRef(new Map<string, number>());

  const loadCatalog = useCallback(async (key: string, signal?: AbortSignal) => {
    if (!key) return;
    const revision = (requests.current.get(key) || 0) + 1;
    requests.current.set(key, revision);
    const payload = await run(async () => {
      try {
        return await apiJson<TestSampleCard[]>(`/test-samples/cards?class_key=${encodeURIComponent(key)}`, { signal });
      } catch (error) { if (!signal?.aborted) throw error; }
    });
    if (payload && !signal?.aborted && requests.current.get(key) === revision) setCards((current) => ({ ...current, [key]: payload }));
    return Boolean(payload);
  }, [run]);

  useEffect(() => {
    if (!classKey || !index) return;
    const controller = new AbortController();
    void loadCatalog(classKey, controller.signal).then(async (loaded) => {
      if (!loaded || controller.signal.aborted) return;
      await run(async () => {
        try {
          await apiJson(`/test-samples/classes/${encodeURIComponent(classKey)}/reconcile`, { method: "POST", signal: controller.signal });
        } catch (error) { if (!controller.signal.aborted) throw error; }
      });
      if (!controller.signal.aborted) void loadCatalog(classKey, controller.signal);
    });
    return () => controller.abort();
  }, [classKey, Boolean(index), loadCatalog, run]);

  const samples = cards[classKey];
  const evaluationActive = samples?.some(
    (sample) => sample.evaluation.status === "queued" || sample.evaluation.status === "running",
  );

  useEffect(() => {
    if (!evaluationActive) return undefined;
    const timer = window.setTimeout(() => void loadCatalog(classKey), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [evaluationActive, loadCatalog, samples, classKey]);

  const removeSample = (sample: TestSampleCard) => {
    showModal({
      title: "Удалить тестовую разметку",
      body: (
        <p>
          Разметка «{sample.name}» и все её файлы будут удалены без возможности восстановления.
          {sample.is_primary ? " Она назначена основной для класса, поэтому тестовый F1 его сетей станет недоступным до назначения новой." : ""}
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
                await Promise.all([loadCatalog(classKey), loadClasses()]);
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
    <div className="test-markup-browser">
      <header className="test-markup-browser-header">
        <div><h1>Тестовые разметки</h1><p>Каталог тестовых разметок по классам</p></div>
        <div className="button-row">
          <a className="primary compact-action" href="#/test-markups/create"><Plus size={15} />Создать</a>
        </div>
      </header>
      <div className="test-markup-browser-layout">
        <nav className="test-markup-class-nav" aria-label="Классы тестовых разметок">
          <span className="field-label">Классы</span>
          {index?.classes?.map((item) => <button key={item.key} type="button" className={classKey === item.key ? "selected" : ""} aria-current={classKey === item.key ? "true" : undefined} onClick={() => setClassKey(item.key)} title={item.name}>
            <span>{item.name}</span><small>{item.sample_count}</small>
          </button>)}
          {!index ? <span className="muted" role="status">Загрузка классов…</span> : null}
        </nav>
        <section className="test-markup-class-content">
          <label className="field test-markup-mobile-class"><span>Класс</span><select aria-label="Класс тестовых разметок" value={classKey} onChange={(event) => setClassKey(event.target.value)} disabled={!index?.classes?.length}>
            {!index?.classes?.length ? <option>{index ? "Нет разметок" : "Загрузка…"}</option> : null}
            {index?.classes?.map((item) => <option key={item.key} value={item.key}>{item.name} · {item.sample_count}</option>)}
          </select></label>
          {classKey && index ? <div className="test-markup-class-header"><h2>{(index.classes || []).find((item) => item.key === classKey)?.name}</h2><span className="muted">Разметок: {samples?.length ?? "…"}</span></div> : null}
          {samples ? <TestSampleCatalog samples={samples} onDelete={removeSample} /> : <div className="empty-state" role="status">{index && !(index.classes || []).length ? "Тестовые разметки ещё не созданы." : "Загрузка разметок выбранного класса…"}</div>}
        </section>
      </div>
    </div>
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

function TestSampleCatalog({
  samples,
  onDelete,
}: {
  samples: TestSampleCard[];
  onDelete: (sample: TestSampleCard) => void;
}) {
  if (!samples.length) return <div className="empty-state">В этом классе пока нет тестовых разметок.</div>;
  return (
    <div className="test-markup-card-grid">
            {samples.map((sample) => (
              <article className="test-markup-card" key={sample.id}>
                <div className="test-markup-card-header">
                  <a href={`#/test-markups/${sample.id}`}>
                    <strong>{sample.name}</strong>
                    <small className="muted">На основе датасета: {sample.source_dataset_name}</small>
                    <small className="muted">
                      Сеть: {sample.source_model_name || "не зафиксирована"}
                      {sample.source_trained_at ? ` · ${formatDateTime(sample.source_trained_at)}` : ""}
                    </small>
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
                  <span><small>F1 пиксельная</small><strong>{formatF1Score(sample.evaluation.pixel?.f1)}</strong></span>
                  <span><small>F1 объектовая</small><strong>{formatF1Score(sample.evaluation.objects?.f1)}</strong></span>
                  <span><small>Тайлы</small><strong>{sample.enabled_image_count}/{sample.image_count}</strong></span>
                </a>
                <CompactPerClassF1
                  metrics={sample.evaluation.metrics}
                  section={sample.quality_metric === "objects" ? "objects" : "pixel"}
                />
                <div className="test-markup-card-footer">
                  <TestSampleEvaluationBadge evaluation={sample.evaluation} />
                  {sample.evaluation.status !== "current" && (sample.evaluation.pixel || sample.evaluation.objects)
                    ? <span className="muted">предыдущие значения</span>
                    : null}
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
}: Pick<RoutedPageProps, "run" | "showModal" | "closeModal" | "registerRouteGuard"> & { sampleId: string }) {
  const [sample, setSample] = useState<TestSampleDetail | null>(null);
  const [draft, setDraft] = useState<TestMarkupDraft | null>(null);
  const [draftEvaluation, setDraftEvaluation] = useState<TestSampleEvaluationInfo | null>(null);
  const [previewPending, setPreviewPending] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [recalculating, setRecalculating] = useState(false);
  const [pseudoLaunching, setPseudoLaunching] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [minTileCount, setMinTileCount] = useState(1);
  const [maxTileCount, setMaxTileCount] = useState(1);
  const [minObjectCount, setMinObjectCount] = useState(1);

  const sampleRequest = useRef(0);
  const loadSample = useCallback(async (signal: AbortSignal) => {
    const revision = ++sampleRequest.current;
    setLoaded(false);
    setSample(null);
    setDraft(null);
    const readSample = () => run(async () => {
      try { return await apiJson<TestSampleDetail>(`/test-samples/${sampleId}`, { signal }); }
      catch (error) { if (!signal.aborted) throw error; }
    });
    const payload = await readSample();
    if (signal.aborted || revision !== sampleRequest.current) return;
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
    if (!payload) return;
    await run(async () => {
      try { await apiJson(`/test-samples/classes/${encodeURIComponent(payload.class_key)}/reconcile`, { method: "POST", signal }); }
      catch (error) { if (!signal.aborted) throw error; }
    });
    if (signal.aborted || revision !== sampleRequest.current) return;
    const refreshed = await readSample();
    if (refreshed && !signal.aborted && revision === sampleRequest.current) setSample(refreshed);
  }, [run, sampleId]);

  useEffect(() => {
    const controller = new AbortController();
    void loadSample(controller.signal);
    return () => { controller.abort(); sampleRequest.current += 1; };
  }, [loadSample]);

  const refreshSample = useCallback(async () => {
    const revision = sampleRequest.current;
    const payload = await run(() => apiJson<TestSampleDetail>(`/test-samples/${sampleId}`));
    if (payload && revision === sampleRequest.current) setSample(payload);
  }, [run, sampleId]);

  const evaluationActive = sample?.evaluation.status === "queued"
    || sample?.evaluation.status === "running";
  const pseudoActive = sample?.pseudo_markup.status === "queued"
    || sample?.pseudo_markup.status === "running";
  useEffect(() => {
    if (!evaluationActive && !pseudoActive) return undefined;
    const timer = window.setTimeout(() => void refreshSample(), PROGRESS_REFRESH_MS);
    return () => window.clearTimeout(timer);
  }, [evaluationActive, pseudoActive, refreshSample, sample]);

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

  const launchPseudoMarkup = async () => {
    setPseudoLaunching(true);
    try {
      const created = await run(() =>
        apiJson<JobDetail>(`/test-samples/${sampleId}/pseudo-markup`, {
          method: "POST",
        }),
      );
      if (created) await refreshSample();
    } finally {
      setPseudoLaunching(false);
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

  const openTilePreview = (tile: NonNullable<TestSampleDetail["tiles"]>[number]) => {
    showModal({
      title: `Тайл ${String(tile.index).padStart(3, "0")} · ${tile.source_name}`,
      fullscreen: true,
      body: (
        <TestSampleTileViewer
          src={tile.preview_url}
          alt={`Полноразмерный тайл ${tile.index}: ${tile.source_name}`}
        />
      ),
    });
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
            ? "После общего сохранения эта разметка заменит текущую основную разметку класса, а оценки сетей всех его датасетов автоматически пересчитаются."
            : "После общего сохранения класс останется без основной тестовой разметки, а тестовые метрики его сетей станут недоступными."}
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
          {sample.is_primary ? " Это основная разметка класса: тестовый F1 всех его сетей станет недоступным." : ""}
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
        subtitle={`${sample.class_name} · создана ${formatDateTime(sample.created_at)}`}
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
          subtitle={`Создана на основе датасета: ${sample.source_dataset_name}${sample.source_dataset_version ? ` · версия ${sample.source_dataset_version}` : ""}`}
          aside={
            <div className="button-row">
              <button className="secondary" type="button" disabled={evaluating || saving || !hasEnabledTiles || sample.pseudo_markup.status !== "ready"} onClick={() => void evaluate()}>
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
          {sample.quality_metric === "objects" ? (
            <Metric
              label="Объекты на границе тайла"
              value={sample.exclude_boundary_objects ? "Исключены" : "Учитываются"}
            />
          ) : null}
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
        <div className="test-sample-evaluation-source">
          <span><strong>Основная сеть:</strong> {sample.pseudo_markup.model_name || "не назначена"}</span>
          <span><strong>Обучающий датасет сети:</strong> {sample.pseudo_markup.training_dataset_name || "не определён"}</span>
          <span>
            <strong>Псевдоразметка исходного датасета:</strong>{" "}
            {{
              ready: "готова",
              queued: "в очереди",
              running: "выполняется",
              unavailable: "отсутствует",
              error: "ошибка",
            }[sample.pseudo_markup.status]}
          </span>
        </div>
        {sample.pseudo_markup.can_create ? (
          <div className="button-row">
            <button className="secondary" type="button" disabled={pseudoLaunching} onClick={() => void launchPseudoMarkup()}>
              <Play size={15} />
              {pseudoLaunching ? "Постановка..." : "Создать псевдоразметку основной сетью"}
            </button>
          </div>
        ) : null}
        {sample.pseudo_markup.job_id && pseudoActive ? (
          <div className="info-box">
            Псевдоразметка формируется. <a href={`#/jobs/${sample.pseudo_markup.job_id}`}>Открыть задание</a>
          </div>
        ) : sample.pseudo_markup.error ? <div className="info-box">{sample.pseudo_markup.error}</div> : null}
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
            <button className="primary" type="submit" disabled={optimizing || !optimizationValid || sample.pseudo_markup.status !== "ready"}>
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
              <button
                className="test-sample-thumbnail"
                type="button"
                aria-label={`Открыть полноразмерный тайл ${tile.index}: ${tile.source_name}`}
                title="Открыть полноразмерный тайл"
                onClick={() => openTilePreview(tile)}
              >
                <img
                  src={tile.thumbnail_url}
                  alt={`Тайл ${tile.index}: ${tile.source_name}`}
                  width="384"
                  height="384"
                  loading="lazy"
                  decoding="async"
                />
                <span className="test-sample-thumbnail-hint"><ZoomIn size={16} />Рассмотреть</span>
              </button>
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

const TILE_VIEWER_MIN_SCALE = 1;
const TILE_VIEWER_MAX_SCALE = 12;
const TILE_VIEWER_ZOOM_STEP = 1.25;

type ViewerPoint = { x: number; y: number };
type ViewerDrag = ViewerPoint & { pointerId: number; origin: ViewerPoint };

function TestSampleTileViewer({ src, alt }: { src: string; alt: string }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<ViewerDrag | null>(null);
  const [scale, setScale] = useState(TILE_VIEWER_MIN_SCALE);
  const [offset, setOffset] = useState<ViewerPoint>({ x: 0, y: 0 });
  const [naturalSize, setNaturalSize] = useState<ViewerPoint | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [viewportSize, setViewportSize] = useState<ViewerPoint>({ x: 0, y: 0 });
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const observer = new ResizeObserver(() => setViewportSize({ x: viewport.clientWidth, y: viewport.clientHeight }));
    observer.observe(viewport);
    return () => observer.disconnect();
  }, []);
  const fittedPixelRatio = naturalSize && viewportSize.x && viewportSize.y
    ? Math.min(1, viewportSize.x / naturalSize.x, viewportSize.y / naturalSize.y) : 1;
  const maxScale = Math.max(TILE_VIEWER_MAX_SCALE, 1 / fittedPixelRatio);

  const resetView = useCallback(() => {
    setScale(TILE_VIEWER_MIN_SCALE);
    setOffset({ x: 0, y: 0 });
  }, []);

  const zoomTo = useCallback((requestedScale: number, clientPoint?: ViewerPoint) => {
    const nextScale = Math.min(
      maxScale,
      Math.max(TILE_VIEWER_MIN_SCALE, requestedScale),
    );
    if (nextScale === scale) return;
    const viewport = viewportRef.current;
    const bounds = viewport?.getBoundingClientRect();
    const anchor = bounds && clientPoint
      ? {
          x: clientPoint.x - bounds.left - bounds.width / 2,
          y: clientPoint.y - bounds.top - bounds.height / 2,
        }
      : { x: 0, y: 0 };
    const ratio = nextScale / scale;
    setOffset((current) => nextScale === TILE_VIEWER_MIN_SCALE
      ? { x: 0, y: 0 }
      : {
          x: anchor.x - (anchor.x - current.x) * ratio,
          y: anchor.y - (anchor.y - current.y) * ratio,
        });
    setScale(nextScale);
  }, [scale, maxScale]);

  const oneToOneScale = useCallback(() => {
    const viewport = viewportRef.current;
    const image = imageRef.current;
    if (!viewport || !image) return TILE_VIEWER_MIN_SCALE;
    return containedImageOneToOneScale(
      viewport.clientWidth,
      viewport.clientHeight,
      image.naturalWidth,
      image.naturalHeight,
      maxScale,
    );
  }, [maxScale]);

  return (
    <div className="test-sample-tile-viewer">
      <div className="test-sample-tile-viewer-toolbar">
        <div className="button-row">
          <button
            className="secondary icon-button compact-action"
            type="button"
            disabled={!loaded || scale <= TILE_VIEWER_MIN_SCALE}
            aria-label="Уменьшить"
            title="Уменьшить"
            onClick={() => zoomTo(scale / TILE_VIEWER_ZOOM_STEP)}
          ><ZoomOut size={17} /></button>
          <span className="test-sample-tile-viewer-scale" title="Масштаб относительно исходных пикселей">{Math.round(scale * fittedPixelRatio * 100)}%</span>
          <button
            className="secondary icon-button compact-action"
            type="button"
            disabled={!loaded || scale >= maxScale}
            aria-label="Увеличить"
            title="Увеличить"
            onClick={() => zoomTo(scale * TILE_VIEWER_ZOOM_STEP)}
          ><ZoomIn size={17} /></button>
          <button className="secondary compact-action" type="button" disabled={!loaded} onClick={() => zoomTo(oneToOneScale())}>1:1</button>
          <button className="secondary compact-action" type="button" disabled={!loaded} onClick={resetView}>Вписать</button>
        </div>
        <span className="muted">
          {naturalSize ? `${naturalSize.x} × ${naturalSize.y} пикс. · ` : ""}
          колесо — масштаб, перетаскивание — перемещение, двойной клик — приблизить
        </span>
      </div>
      <div
        ref={viewportRef}
        className={`test-sample-tile-viewer-viewport${dragging ? " dragging" : ""}`}
        role="region"
        aria-label="Просмотр полноразмерного тайла"
        tabIndex={0}
        onWheel={(event) => {
          event.preventDefault();
          zoomTo(
            scale * (event.deltaY < 0 ? TILE_VIEWER_ZOOM_STEP : 1 / TILE_VIEWER_ZOOM_STEP),
            { x: event.clientX, y: event.clientY },
          );
        }}
        onDoubleClick={(event) => {
          if (scale > TILE_VIEWER_MIN_SCALE) resetView();
          else zoomTo(Math.max(2, oneToOneScale()), { x: event.clientX, y: event.clientY });
        }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = {
            pointerId: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            origin: offset,
          };
          setDragging(true);
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || drag.pointerId !== event.pointerId) return;
          setOffset({
            x: drag.origin.x + event.clientX - drag.x,
            y: drag.origin.y + event.clientY - drag.y,
          });
        }}
        onPointerUp={(event) => {
          if (dragRef.current?.pointerId !== event.pointerId) return;
          dragRef.current = null;
          setDragging(false);
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          dragRef.current = null;
          setDragging(false);
        }}
        onKeyDown={(event) => {
          if (event.key === "+" || event.key === "=") zoomTo(scale * TILE_VIEWER_ZOOM_STEP);
          if (event.key === "-") zoomTo(scale / TILE_VIEWER_ZOOM_STEP);
          if (event.key === "0") resetView();
        }}
      >
        {!loaded && !failed ? <div className="test-sample-tile-viewer-status">Загрузка полноразмерного тайла…</div> : null}
        {failed ? <div className="test-sample-tile-viewer-status error-text">Не удалось загрузить полноразмерный тайл.</div> : null}
        <img
          ref={imageRef}
          src={src}
          alt={alt}
          draggable={false}
          decoding="async"
          style={{
            width: naturalSize ? naturalSize.x * fittedPixelRatio * scale : undefined,
            height: naturalSize ? naturalSize.y * fittedPixelRatio * scale : undefined,
            transform: `translate3d(${offset.x}px, ${offset.y}px, 0) translate(-50%, -50%)`,
          }}
          onLoad={(event) => {
            setNaturalSize({ x: event.currentTarget.naturalWidth, y: event.currentTarget.naturalHeight });
            setLoaded(true);
            setFailed(false);
            resetView();
          }}
          onError={() => {
            setLoaded(false);
            setFailed(true);
          }}
        />
      </div>
    </div>
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
                {recalculating ? "Постановка..." : "Пересчитать основной сетью класса"}
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
      <PerClassF1Table metrics={evaluation.metrics} />
      <div className="test-sample-evaluation-source">
        <span>
          <strong>{direct ? "Рассчитано сетью:" : "Псевдоразметка сети:"}</strong>{" "}
          {evaluation.model_name || (direct ? "ещё не рассчитано" : "нет подходящей псевдоразметки")}
          {evaluation.training_dataset_name ? ` · датасет обучения: ${evaluation.training_dataset_name}` : ""}
          {direct && evaluation.trained_at ? ` · обучение: ${formatDateTime(evaluation.trained_at)}` : ""}
        </span>
        {direct && evaluation.target_model_name && evaluation.target_model_name !== evaluation.model_name ? (
          <span>
            <strong>Текущая основная сеть:</strong> {evaluation.target_model_name}
            {evaluation.target_training_dataset_name ? ` · датасет обучения: ${evaluation.target_training_dataset_name}` : ""}
            {evaluation.target_trained_at ? ` · обучение: ${formatDateTime(evaluation.target_trained_at)}` : ""}
          </span>
        ) : null}
        {!direct && evaluation.markup_created_at ? <span><strong>Псевдоразметка:</strong> {formatDateTime(evaluation.markup_created_at)}</span> : null}
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
  const [userDrafts, setUserDrafts] = useState<DatasetEditorUserDraftInfo[] | null>(null);

  const loadCatalog = useCallback(async () => {
    const payload = await run(() => apiJson<DatasetCatalogInfo>("/dataset-catalog"));
    if (payload) setCatalog(payload);
  }, [run]);

  const loadUserDrafts = useCallback(async () => {
    const payload = await run(() =>
      apiJson<DatasetEditorUserDraftListResponse>("/dataset-editor/drafts"),
    );
    if (payload) setUserDrafts(payload.drafts);
  }, [run]);

  useEffect(() => {
    void loadCatalog();
    void loadUserDrafts();
  }, [loadCatalog, loadUserDrafts]);

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
    const technicalName = String(data.get("technical_name") || "").trim().toLowerCase();
    const imageryType = String(data.get("imagery_type") || "kanopus") as ImageryType;
    if (!name || !technicalName) return;
    const payload = await run(() =>
      apiJson<DatasetCatalogInfo>("/dataset-classes", {
        method: "POST",
        body: { name, technical_name: technicalName, imagery_type: imageryType },
      }),
    );
    if (payload) form.reset();
    await applyCatalog(payload);
  };

  const renameClass = async (event: FormEvent<HTMLFormElement>, classKey: string) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const name = String(data.get("name") || "").trim();
    const technicalName = String(data.get("technical_name") || "").trim().toLowerCase();
    await applyCatalog(await run(() =>
      apiJson<DatasetCatalogInfo>(`/dataset-classes/${encodeURIComponent(classKey)}`, {
        method: "PATCH",
        body: { name, technical_name: technicalName },
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

  const openManagedDatasetParameters = (
    classInfo: NonNullable<DatasetCatalogInfo["classes"]>[number],
    dataset?: DatasetInfo,
  ) => {
    if (!catalog) return;
    showModal({
      title: dataset
        ? `Управляемый датасет «${dataset.dataset_name || dataset.name}»`
        : "Новый управляемый датасет",
      wide: true,
      body: (
        <ManagedDatasetForm
          targetClass={classInfo}
          catalog={catalog}
          dataset={dataset}
          run={run}
          onCancel={closeModal}
          onSaved={async (payload) => {
            closeModal();
            await applyCatalog(payload);
          }}
        />
      ),
    });
  };

  const openDatasetCopy = (
    classInfo: NonNullable<DatasetCatalogInfo["classes"]>[number],
    dataset: DatasetInfo,
  ) => {
    const sourceName = dataset.dataset_name || dataset.name;
    const existingNames = new Set(
      (classInfo.datasets || []).map((item) => (
        item.dataset_name || item.name
      ).toLocaleLowerCase("ru")),
    );
    let suggestedName = `${sourceName} копия`;
    let suffix = 2;
    while (existingNames.has(suggestedName.toLocaleLowerCase("ru"))) {
      suggestedName = `${sourceName} копия ${suffix}`;
      suffix += 1;
    }
    const formId = `dataset-copy-${dataset.key}`;
    showModal({
      title: `Создать копию датасета «${sourceName}»`,
      body: (
        <form
          id={formId}
          className="form-stack"
          onSubmit={async (event) => {
            event.preventDefault();
            const data = new FormData(event.currentTarget);
            const name = String(data.get("name") || "").trim();
            if (!name) return;
            const result = await run(() => apiJson<DatasetEditorCopyResult>(
              `/dataset-editor/datasets/${encodeURIComponent(dataset.key)}/copy`,
              { method: "POST", body: { name } },
            ));
            if (!result) return;
            closeModal();
            await loadCatalog();
            await reloadBootstrap();
            showModal({
              title: "Копия датасета создана",
              body: result.managed ? (
                <p>
                  Управляемый датасет <strong>{result.dataset_name}</strong> создан с теми же
                  источниками, приоритетами и добавленными снимками.
                </p>
              ) : (
                <p>
                  Создан датасет <strong>{result.dataset_name}</strong> и Git-коммит{" "}
                  <strong>{result.commit.slice(0, 8)}</strong>. Публикация MLMarkup:
                  {result.publication_status === "published" ? " завершена" : " выполняется"}.
                </p>
              ),
              footer: (
                <button className="primary" type="button" onClick={closeModal}>Готово</button>
              ),
            });
          }}
        >
          <label>
            Название нового датасета
            <input name="name" defaultValue={suggestedName} maxLength={240} autoFocus required />
          </label>
          <div className="notice">
            {dataset.managed ? (
              <>
                Копируются параметры управляемого датасета. Исходные датасеты остаются общими,
                задания, результаты обучения и черновики не переносятся.
              </>
            ) : (
              <>
                Копируются опубликованные файлы разметки в новую независимую папку MLMarkup.
                Черновики, задания, шаблоны и результаты обучения не переносятся.
              </>
            )}
          </div>
        </form>
      ),
      footer: (
        <>
          <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
          <button className="primary" type="submit" form={formId}>
            <Copy size={15} /> Создать копию
          </button>
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
            {dataset.managed ? (
              <>Управляемый датасет будет удалён из каталога. Исходные датасеты и их разметка останутся.</>
            ) : (
              <>Папка <strong>{dataset.source_path}</strong> будет удалена из MLMarkup отдельным Git-коммитом.</>
            )}
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
              await loadUserDrafts();
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
      {userDrafts?.length ? (
        <section className="panel class-editor-user-drafts">
          <PanelHeader
            title="Мои черновики"
            subtitle="Сохранённые, но ещё не опубликованные изменения"
          />
          <div className="class-editor-user-draft-list">
            {userDrafts.map((draft) => (
              <a
                className="class-editor-user-draft"
                href={`#/dataset-editor/${encodeURIComponent(draft.dataset_key)}`}
                key={draft.dataset_key}
              >
                <FileText size={18} />
                <span className="source-lines">
                  <strong>{draft.class_name} · {draft.dataset_name}</strong>
                  <span className="muted">
                    Снимков с изменениями: {draft.scene_count}
                    {draft.deleted_scene_count
                      ? ` · на удаление: ${draft.deleted_scene_count}`
                      : ""}
                  </span>
                  <span className="muted">Обновлено: {formatDateTime(draft.updated_at)}</span>
                </span>
                <span className="class-editor-user-draft-action">
                  Продолжить <ExternalLink size={14} />
                </span>
              </a>
            ))}
          </div>
        </section>
      ) : null}
      <section className="panel">
        <PanelHeader title="Новый класс" subtitle="Выберите тип снимков, затем добавьте датасеты" />
        <form className="inline-form" onSubmit={createClass}>
          <input name="name" placeholder="Название класса" maxLength={240} required />
          <input
            name="technical_name"
            placeholder="Техническое имя, например abrasion"
            maxLength={160}
            pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
            title="Латинские строчные буквы, цифры, дефис и подчёркивание"
            required
          />
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
                <input
                  name="technical_name"
                  defaultValue={classInfo.technical_name}
                  maxLength={160}
                  pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
                  title="Техническое имя модели: латинские строчные буквы, цифры, дефис и подчёркивание"
                  aria-label="Техническое имя класса"
                  required
                />
                <button className="secondary" type="submit">Сохранить</button>
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
                      {dataset.managed ? (
                        <>
                          <span className="badge neutral">управляемый</span>
                          <span className="muted">
                            Источники: {(dataset.managed_sources || []).map((source) => (
                              `${source.class_name}\\${source.dataset_name} · ${source.priority}`
                            )).join("; ")}
                          </span>
                        </>
                      ) : <span className="muted">MLMarkup: {dataset.source_path}</span>}
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
                        onClick={() => dataset.managed
                          ? openManagedDatasetParameters(classInfo, dataset)
                          : openDatasetEditor(classInfo.key, dataset)}
                      >
                        Параметры
                      </button>
                      <button
                        className="secondary icon-button"
                        type="button"
                        aria-label={`Создать копию датасета ${dataset.dataset_name || dataset.name}`}
                        title="Создать копию под новым именем"
                        onClick={() => openDatasetCopy(classInfo, dataset)}
                      >
                        <Copy size={15} />
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
              <button
                className="secondary"
                type="button"
                onClick={() => openManagedDatasetParameters(classInfo)}
              >
                <Layers3 size={15} /> Создать управляемый
              </button>
            </div>
          </section>
        );
      })}
    </>
  );
}

function TemplatesPage({ bootstrap, run, reloadBootstrap, showModal, closeModal }: RoutedPageProps) {
  const [visibleMode, setVisibleMode] = useState<"training" | "inference">(currentRoute()[1] === "inference" ? "inference" : "training");
  const [trainingId, setTrainingId] = useState(bootstrap.training_templates[0]?.id || "");
  const trainingTemplate = byId(bootstrap.training_templates, trainingId) || bootstrap.training_templates[0];
  const [trainingConfig, setTrainingConfig] = useState<JsonRecord>({});
  useEffect(() => setTrainingConfig({ ...(trainingTemplate?.default_config || {}) }), [trainingTemplate?.id]);
  const save = async () => {
    if (!trainingTemplate) return false;
    const updated = await run(() => apiJson<TrainingTemplate>(`/training-templates/by-id/${trainingTemplate.id}`, { method: "PUT", body: { default_config: trainingConfig } }));
    if (!updated) return false;
    await reloadBootstrap();
    return true;
  };
  const reset = async () => {
    if (!trainingTemplate) return;
    const updated = await run(() => apiJson<TrainingTemplate>(`/training-templates/by-id/${trainingTemplate.id}`, { method: "PUT", body: { reset_to_baseline: true } }));
    if (updated) { setTrainingConfig({ ...updated.default_config }); await reloadBootstrap(); }
  };
  const remove = () => {
    if (!trainingTemplate) return;
    showModal({ title: "Удалить шаблон", body: <p>{trainingTemplate.display_name}</p>, footer: <>
      <button className="secondary" type="button" onClick={closeModal}>Отмена</button>
      <button className="danger" type="button" onClick={async () => {
        const deleted = await run(() => apiJson(`/training-templates/by-id/${trainingTemplate.id}`, { method: "DELETE" }));
        if (deleted) { closeModal(); await reloadBootstrap(); }
      }}>Удалить</button></> });
  };
  const help = () => showModal({
    title: "Как работают шаблоны",
    body: <div className="form-stack templates-help">
      <div><h3>Обучение</h3><p>Базовый шаблон задаёт параметры сети. Шаблон для конкретного датасета уточняет их и используется вместо базового. При запуске выбранные параметры сохраняются в задании.</p></div>
      <div><h3>Инференс</h3><p>Именованный шаблон задаёт обработку прогнозов: фильтры маски, минимальную площадь объектов, сглаживание и упрощение контуров. Его можно назначить нескольким классам независимо от сети, датасета и типа снимков. У каждого класса один шаблон.</p></div>
      <div><h3>Назначение классов</h3><p>Нажмите на плашку класса, чтобы перенести его в другой шаблон или снять привязку. Создание шаблона не назначает классы автоматически. После удаления шаблона его классы остаются без назначения; создание псевдоразметки требует нового назначения.</p></div>
      <div><h3>На что влияют изменения</h3><p>Параметры инференса используются при создании псевдоразметки, обработке области, оценке F1 на тестовой разметке и экспорте модели. Изменения действуют для будущих операций всех привязанных классов. Запущенные задания и готовые псевдоразметки сохраняют свои параметры. Обучение сети от шаблона инференса не зависит.</p></div>
    </div>,
    footer: <button type="button" className="secondary" onClick={closeModal}>Понятно</button>,
  });
  const header = (create?: () => void) => <header className="templates-header">
      <div className="templates-heading"><h1>Шаблоны</h1><button type="button" className="templates-help-button" onClick={help} title="Как работают шаблоны" aria-label="Как работают шаблоны" aria-haspopup="dialog"><CircleHelp size={19} /></button></div>
      <div className="template-mode-tabs" role="group" aria-label="Тип шаблонов">
        <button type="button" className={visibleMode === "training" ? "primary" : "secondary"} aria-pressed={visibleMode === "training"} onClick={() => { setVisibleMode("training"); usagePageViewed("training_templates"); }}>Обучение</button>
        <button type="button" className={visibleMode === "inference" ? "primary" : "secondary"} aria-pressed={visibleMode === "inference"} onClick={() => { setVisibleMode("inference"); usagePageViewed("inference_templates"); }}>Инференс</button>
      </div>
      {create ? <button type="button" className="primary templates-create-button" onClick={create}><Plus size={16} />Новый шаблон</button> : null}
    </header>;
  return <div className="templates-page">
    {visibleMode === "inference" ? <InferenceTemplates bootstrap={bootstrap} run={run} reload={reloadBootstrap} showModal={showModal} closeModal={closeModal} renderHeader={header} /> : <>
      {header()}
      <section className="two-column templates-layout" data-visible-mode="training">
        <TemplateTree mode="training" title="Шаблоны обучения" templates={bootstrap.training_templates} selectedId={trainingTemplate?.id || ""} onSelect={setTrainingId}
          onAdd={() => showModal({ title: "Добавить шаблон обучения", body: <CreateTemplateForm models={bootstrap.models} datasets={bootstrap.datasets} templates={bootstrap.training_templates} run={run} closeModal={closeModal} reloadBootstrap={reloadBootstrap} /> })} />
        {trainingTemplate ? <TemplateEditor key={trainingTemplate.id} mode="training" template={trainingTemplate} config={trainingConfig} onConfig={setTrainingConfig} onSave={save} onReset={() => void reset()} onDelete={trainingTemplate.dataset_key ? remove : undefined} /> : null}
      </section></>}
  </div>;
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

function openTrainingStopModal(
  job: JobSummary | JobDetail,
  run: Runner,
  showModal: (modal: ModalState) => void,
  closeModal: () => void,
  onChanged: () => Promise<void>,
) {
  const checkpointAvailable = Boolean(job.best_checkpoint_available);
  showModal({
    title: "Остановить обучение?",
    wide: true,
    body: (
      <div className="form-stack">
        <p>
          Можно завершить обучение штатно и сохранить как успешный результат <strong>best.pt</strong> —
          {" чекпойнт эпохи с максимальной валидационной F1 по метрике этого обучения."}
        </p>
        <p>Текущая незавершённая эпоха будет отброшена. Файл final.pt вместо лучшего не используется.</p>
        {!checkpointAvailable ? (
          <div className="notice warning">
            Лучший чекпойнт появится после первой полностью завершённой эпохи. Сейчас доступна только
            остановка без результата.
          </div>
        ) : null}
      </div>
    ),
    footer: (
      <>
        <button className="secondary" type="button" onClick={closeModal}>Продолжить обучение</button>
        <button
          className="primary"
          type="button"
          disabled={!checkpointAvailable}
          title={checkpointAvailable ? "Сохранить чекпойнт с лучшей F1" : "Первая эпоха ещё не завершена"}
          onClick={async () => {
            const updated = await run(() => apiJson<JobDetail>(`/jobs/${job.id}/stop-and-save-best`, { method: "POST" }));
            if (!updated) return;
            closeModal();
            await onChanged();
          }}
        >
          <Save size={16} /> Сохранить лучший и остановить
        </button>
        <button
          className="danger"
          type="button"
          onClick={async () => {
            const deleted = await run(() => apiJson<JobDetail>(`/jobs/${job.id}`, { method: "DELETE" }));
            if (!deleted) return;
            closeModal();
            await onChanged();
          }}
        >
          <Trash2 size={16} /> Остановить без результата
        </button>
      </>
    ),
  });
}

function QueuePage({ run, showModal, closeModal }: RoutedPageProps) {
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
    if (
      action === "delete"
      && job.type === "training"
      && job.source === "manual"
      && isActiveStatus(job.status)
    ) {
      openTrainingStopModal(job, run, showModal, closeModal, load);
      return;
    }
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

function JobPage({ bootstrap, run, showModal, closeModal, jobId }: RoutedPageProps & { jobId: string }) {
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
        subtitle={`${job.dataset_name} · ${trainingModelLabel(job.model_name, job.pipeline_variant)}`}
        actions={
          <div className="inline-row">
            {job.type === "training" && job.dataset_key ? (
              <a className="secondary" href={`#/results/${encodeURIComponent(job.dataset_key)}`}>
                <BarChart3 size={16} />
                Результаты обучения
              </a>
            ) : null}
            {job.type === "training" ? (
              job.mlflow_run_url ? (
                <a className="secondary" href={job.mlflow_run_url} target="_blank" rel="noreferrer">
                  <ExternalLink size={16} />
                  Открыть MLflow
                </a>
              ) : (
                <button className="secondary" type="button" disabled title="Запуск MLflow ещё не создан">
                  <ExternalLink size={16} />
                  Открыть MLflow
                </button>
              )
            ) : null}
            {job.type === "training" && job.source === "manual" && isActiveStatus(job.status) ? (
              <button
                className="danger"
                type="button"
                disabled={job.stop_and_save_best_requested}
                onClick={() => openTrainingStopModal(job, run, showModal, closeModal, load)}
              >
                <Square size={15} />
                {job.stop_and_save_best_requested ? "Останавливается" : "Остановить"}
              </button>
            ) : null}
            <a className="secondary" href="#/queue">К очереди</a>
          </div>
        }
      />
      <section className="panel">
        <div className="metric-grid">
          <Metric
            label="Статус"
            value={job.stop_and_save_best_requested && isActiveStatus(job.status)
              ? <span className="badge warning">сохраняется лучший чекпойнт по F1</span>
              : statusBadge(job.status, job.type, job.progress)}
          />
          <Metric label="Тип" value={job.purpose === "test_sample_f1" ? "тестовый F1" : job.purpose === "pseudo_markup" ? "разметка" : "обучение"} />
          <Metric label="Источник" value={sourceBadge(job.source)} />
          <Metric label="Приоритет" value={job.secondary_priority ? "второстепенный" : "обычный"} />
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

function ResultsPage({ run, showJobLog }: Pick<RoutedPageProps, "run" | "showJobLog">) {
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
  getBootstrap,
  run,
  showModal,
  closeModal,
  showJobLog,
}: Pick<RoutedPageProps, "run" | "showModal" | "closeModal" | "showJobLog"> & {
  datasetKey: string;
  bootstrap: BootstrapInfo | null;
  getBootstrap: () => Promise<BootstrapInfo | undefined>;
}) {
  const [payload, setPayload] = useState<DatasetResultsResponse | null>(null);
  const [recalculatingTestF1, setRecalculatingTestF1] = useState(false);
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

  const showPseudo = async (result: TrainingResultInfo) => {
    const catalog = await getBootstrap();
    if (!catalog) return;
    showModal({
      title: "Создать псевдоразметку",
      body: (
        <PseudoMarkupForm
          datasetKey={datasetKey}
          result={result}
          datasets={catalog.datasets}
          imageFolders={catalog.image_folders}
          inferenceAvailable={Boolean(inferenceTemplateForDataset(catalog.inference_templates, catalog.datasets, payload.class_key || datasetKey))}
          run={run}
          closeModal={closeModal}
          reload={load}
        />
      ),
    });
  };

  const showZip = async (result: TrainingResultInfo) => {
    const catalog = await getBootstrap();
    if (catalog) showTrainingResultZipModal(result, catalog.datasets, run, showModal, closeModal);
  };

  const showContinuation = async (result: TrainingResultInfo) => {
    const [options, catalog] = await Promise.all([
      run(() => apiJson<TrainingContinuationOptions>(`/results/training/${result.id}/continue`)),
      getBootstrap(),
    ]);
    if (options && catalog) showModal({
      title: "Продолжить обучение",
      body: <TrainingContinuationForm inferenceAvailable={Boolean(inferenceTemplateForDataset(catalog.inference_templates, catalog.datasets, payload.class_key || datasetKey))} result={result} options={options} run={run} closeModal={closeModal} reload={load} />,
      footer: null,
    });
  };

  const togglePrimaryResult = async (result: TrainingResultInfo) => {
    const updated = await run(() =>
      apiJson<TrainingResultInfo>(`/results/training/${result.id}/primary`, {
        method: result.is_primary ? "DELETE" : "POST",
      }),
    );
    if (updated) await load();
  };

  const deletePseudo = (item: PseudoMarkupResultInfo) => {
    showModal({
      title: "Удалить pseudo-markup",
      body: <p>{imageSourceLabel(item, bootstrap?.datasets || [], bootstrap?.image_folders || [])}</p>,
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
    if (recalculatingTestF1) return;
    setRecalculatingTestF1(true);
    try {
      const updated = await run(() =>
        apiJson<DatasetResultsResponse>(`/results/datasets/${encodeURIComponent(datasetKey)}/test-f1`, {
          method: "POST",
        }),
      );
      if (updated) setPayload(updated);
    } finally {
      setRecalculatingTestF1(false);
    }
  };
  const primaryTestSamples = payload.primary_test_samples?.length
    ? payload.primary_test_samples
    : payload.primary_test_sample
      ? [payload.primary_test_sample]
      : [];
  const managedTestEvaluation = primaryTestSamples.some((sample) => sample.class_id !== null && sample.class_id !== undefined);
  const resultMetricLabel = managedTestEvaluation ? "F1 сред." : qualityMetricShort(payload.quality_metric);

  return (
    <>
      <PageHeader
        title={payload.dataset_name}
        subtitle={`Обновление датасета: ${formatDate(payload.dataset_updated_at)}`}
        actions={
          <>
            <a
              className="secondary"
              href={`#/dataset-editor/${encodeURIComponent(datasetKey)}`}
            >
              <PencilLine size={16} />
              Редактор датасета
            </a>
            <a className="secondary" href="#/results">Все классы</a>
          </>
        }
      />
      {primaryTestSamples.length ? (
        <section className={`status-banner ${payload.test_f1_status === "current" ? "ok" : payload.test_f1_status === "running" ? "neutral" : "error"}`}>
          <div>
            <strong>{payload.test_f1_status === "current" ? `${resultMetricLabel} актуален` : payload.test_f1_status === "running" ? `Идёт пересчёт ${resultMetricLabel}` : `${resultMetricLabel} не актуален`}</strong>
            <span className="result-test-sample-list">
              {primaryTestSamples.map((sample) => (
                <span className="result-test-sample" key={sample.id} title={`Основная тестовая разметка: ${sample.name}`}>
                  {sample.color ? <span className="class-color-dot" style={{ backgroundColor: sample.color }} /> : null}
                  <strong>{sample.class_name || "Основная разметка"}</strong>
                  <span>{sample.name}</span>
                  <small>{sample.enabled_image_count} тайлов</small>
                </span>
              ))}
            </span>
          </div>
          <button className="primary" type="button" disabled={recalculatingTestF1} onClick={() => void recalculateTestF1()}>
            <RefreshCw size={16} />
            {recalculatingTestF1 ? "Постановка в очередь..." : "Пересчитать по всем сетям датасета"}
          </button>
        </section>
      ) : (
        <section className="status-banner neutral"><strong>Основная тестовая разметка не назначена</strong></section>
      )}
      <section className="panel">
        <ResultsTable
          payload={payload}
          datasets={bootstrap?.datasets || []}
          imageFolders={bootstrap?.image_folders || []}
          onPseudo={(result) => void showPseudo(result)}
          onZip={(result) => void showZip(result)}
          onContinue={(result) => void showContinuation(result)}
          onPrimary={(result) => void togglePrimaryResult(result)}
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

function TemplateTree({
  mode,
  title,
  templates,
  selectedId,
  onSelect,
  onAdd,
}: {
  mode: "training";
  title: string;
  templates: TrainingTemplate[];
  selectedId: string;
  onSelect: (id: string) => void;
  onAdd: () => void;
}) {
  const bases = templates.filter((item) => !item.dataset_key);
  return (
    <section className="panel template-tree-panel" data-template-mode={mode}>
      <PanelHeader
        title={title}
        aside={
          <button className="secondary compact-action" type="button" onClick={onAdd}>
            <Plus size={14} />
            Добавить
          </button>
        }
      />
      <label className="field mobile-template-picker">
        <span>Выбранный шаблон</span>
        <select value={selectedId} onChange={(event) => onSelect(event.target.value)}>
          {templates.map((template) => <option key={template.id} value={template.id}>{templateTitle(template)}</option>)}
        </select>
      </label>
      <div className="template-tree">
        {bases.map((base) => (
          <div key={base.id}>
            <TreeButton template={base} active={base.id === selectedId} onClick={() => onSelect(base.id)} />
            {templates
              .filter((item) => item.architecture === base.architecture && item.dataset_key)
              .sort((a, b) => templateTitle(a).localeCompare(templateTitle(b), "ru"))
              .map((child) => (
                <TreeButton child template={child} active={child.id === selectedId} onClick={() => onSelect(child.id)} key={child.id} />
              ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function TreeButton({ template, active, child = false, onClick }: { template: TrainingTemplate; active: boolean; child?: boolean; onClick: () => void }) {
  return (
    <button className={`tree-button ${child ? "child" : ""} ${active ? "active" : ""}`} type="button" onClick={onClick}>
      <span>{templateTitle(template)}</span>
      <span className={`badge ${template.source === "manual" ? "warning" : "ok"}`}>{template.source === "manual" ? "Настроен" : "Исходный"}</span>
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
}: {
  mode: "training";
  template: TrainingTemplate;
  config: JsonRecord;
  onConfig: (next: JsonRecord) => void;
  onSave: () => Promise<boolean>;
  onReset: () => void;
  onDelete?: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const [savedTemplateName, setSavedTemplateName] = useState<string | null>(null);
  const save = async () => {
    if (saving) return;
    setSaving(true);
    setSavedTemplateName(null);
    try {
      if (await onSave()) setSavedTemplateName(templateTitle(template));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="panel template-editor-panel" data-template-mode={mode}>
      <PanelHeader
        title={`Обучение: ${templateTitle(template)}`}
        subtitle={template.dataset_key ? "Шаблон датасета" : "Базовый шаблон сети"}
        aside={<span className="badge neutral">Версия {template.version}</span>}
      />
      <ConfigEditor
        schema={template.config_schema}
        value={config}
        onChange={(next) => { setSavedTemplateName(null); onConfig(next); }}
        readonly={saving}
        architecture={template.architecture}
      />
      <div className="button-row">
        <button className="primary" type="button" disabled={saving} aria-busy={saving} onClick={() => void save()}>
          {saving ? <LoaderCircle className="status-spinner" size={16} aria-hidden="true" /> : null}
          {saving ? "Сохранение…" : "Сохранить"}
        </button>
        <button className="secondary" type="button" disabled={saving} onClick={() => { setSavedTemplateName(null); onReset(); }}>
          Сбросить
        </button>
        {onDelete ? (
          <button className="danger" type="button" disabled={saving} onClick={onDelete}>
            <Trash2 size={16} />
            Удалить
          </button>
        ) : null}
      </div>
      {savedTemplateName ? <p className="template-save-status" role="status">Сохранён шаблон «{savedTemplateName}».</p> : null}
    </section>
  );
}

function CreateTemplateForm({
  models,
  datasets,
  templates,
  run,
  closeModal,
  reloadBootstrap,
}: {
  models: Pick<ModelInfo, "architecture" | "display_name">[];
  datasets: DatasetInfo[];
  templates: TrainingTemplate[];
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
    const created = await run(() =>
      apiJson<TrainingTemplate>("/training-templates", {
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
      <table className="queue-table">
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
              <td className="technical-value" data-label="В очереди">{job.queue_position}</td>
              <td>
                {job.stop_and_save_best_requested && isActiveStatus(job.status)
                  ? <span className="badge warning">сохраняется лучший F1</span>
                  : statusBadge(job.status, job.type, job.progress)}
              </td>
              <td>
                <span className="inline-row">
                  {jobTypeBadge(job)}
                  {job.secondary_priority ? <span className="badge neutral">второстепенное</span> : null}
                </span>
              </td>
              <td data-label="Датасет">{queueDatasetCell(job)}</td>
              <td data-label="Модель">{queueModelCell(job)}</td>
              <td className="technical-value" data-label="Создано">{formatDateTime(job.created_at)}</td>
              <td>
                <div className="inline-row" onClick={(event) => event.stopPropagation()}>
                  <a className="secondary compact-action" href={`#/jobs/${job.id}`}>
                    Job
                  </a>
                  <button className="secondary icon-button" type="button" title="Выше" disabled={!(job.actions || []).includes("move_up")} onClick={() => onAction(job, "move-up")}>
                    <ChevronUp size={15} />
                  </button>
                  <button className="secondary icon-button" type="button" title="Ниже" disabled={!(job.actions || []).includes("move_down")} onClick={() => onAction(job, "move-down")}>
                    <ChevronDown size={15} />
                  </button>
                  <button
                    className="danger icon-button"
                    type="button"
                    disabled={job.stop_and_save_best_requested}
                    title={job.type === "training" && isActiveStatus(job.status) ? "Остановить обучение" : "Удалить"}
                    onClick={() => onAction(job, "delete")}
                  >
                    {job.type === "training" && isActiveStatus(job.status) ? <Square size={15} /> : <Trash2 size={15} />}
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
      <div className="dataset-list">
        {datasets.length ? (
          datasets.map((dataset) => (
            <div
              className={`dataset-link${dataset.is_primary ? " primary-dataset" : ""}`}
              key={dataset.key}
              title={dataset.is_primary ? "Основной датасет класса" : undefined}
            >
              <div className="dataset-result-link">
                <a className="dataset-result-identity" href={`#/results/${encodeURIComponent(dataset.key)}`}>
                  <span className="dataset-result-name">{dataset.dataset_name || dataset.name}</span>
                  <small>{integerOrNull(dataset.image_count) ?? "—"} снимков</small>
                </a>
                {dataset.test_f1 !== null && dataset.test_f1 !== undefined ? (
                  <span className="dataset-result-score">
                    <a
                      className={`result-card-f1 test-f1-link ${dataset.test_f1_status === "current" ? "current" : "stale"}`}
                      href={dataset.test_f1_training_result_id ? `#/test-f1/${dataset.test_f1_training_result_id}` : `#/results/${encodeURIComponent(dataset.key)}`}
                      title="Открыть снимки с эталоном, прогнозом и тестовым F1 этой сети"
                      aria-label={`Посмотреть тестовый F1 датасета ${dataset.dataset_name || dataset.name}`}
                    >
                      <small>{metricAggregationLabel(dataset.test_f1_metrics, dataset.quality_metric)}</small>
                      {formatTestF1Percent(dataset.test_f1)}
                    </a>
                    <CompactPerClassF1
                      metrics={dataset.test_f1_metrics}
                      section={dataset.quality_metric === "objects" ? "objects" : "pixel"}
                    />
                  </span>
                ) : null}
              </div>
              <a
                className="dataset-editor-link"
                href={`#/dataset-editor/${encodeURIComponent(dataset.key)}`}
                title="Открыть редактор датасета"
                aria-label={`Открыть редактор датасета ${dataset.dataset_name || dataset.name}`}
              >
                <PencilLine size={15} />
              </a>
            </div>
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
  onContinue,
  onPrimary,
  onDeletePseudo,
  showJobLog,
}: {
  payload: DatasetResultsResponse;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  onPseudo: (result: TrainingResultInfo) => void;
  onZip: (result: TrainingResultInfo) => void;
  onContinue: (result: TrainingResultInfo) => void;
  onPrimary: (result: TrainingResultInfo) => void;
  onDeletePseudo: (item: PseudoMarkupResultInfo) => void;
  showJobLog: (jobId: string) => Promise<void>;
}) {
  if (!payload.results.length) return <div className="empty-state">Для датасета пока нет результатов</div>;
  return (
    <div className="form-stack">
      {trainingResultFamilies(payload.results).map((family) => (
        <section className={family.stages.length > 1 ? "training-family" : undefined} key={family.id}>
          {family.stages.length > 1 ? <header className="training-family-heading"><strong>Связанные этапы обучения</strong><span className="muted">У каждого этапа свои веса и псевдоразметки</span></header> : null}
          {family.stages.map((result, stageIndex) => (
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
                      <strong className="inline-row">
                        {result.status === "ok" ? (
                          <button
                            className="icon-button primary-result-star"
                            type="button"
                            title={result.is_primary ? "Снять отметку основной сети класса" : "Сделать основной сетью класса"}
                            aria-label={result.is_primary ? "Снять отметку основной сети класса" : "Сделать основной сетью класса"}
                            onClick={() => onPrimary(result)}
                          >
                            <Star className={result.is_primary ? "primary-star" : undefined} size={17} fill={result.is_primary ? "currentColor" : "none"} />
                          </button>
                        ) : null}
                        {trainingModelLabel(result.model_name, result.pipeline_variant)}
                      </strong>
                      <small className="muted">{result.architecture}</small>
                      {family.stages.length > 1 ? <small className="training-stage-label">Этап {stageIndex + 1}{!result.continued_from_result_id ? " · исходная сеть" : ""}</small> : null}
                      {result.continued_from_result_id ? <small className="muted">{family.stages.some(stage => stage.id === result.continued_from_result_id) ? `От этапа ${family.stages.findIndex(stage => stage.id === result.continued_from_result_id) + 1}` : "Продолжение"} · {result.continued_from_checkpoint === "last" ? "последние веса" : `лучшие веса${result.continued_from_epoch != null ? ` эпохи ${result.continued_from_epoch}` : ""}`}</small> : null}
                    </span>
                  </td>
                  <td title="Статус">
                    <span className="status-stack">
                      {resultStatusBadge(result.status, "training", result.progress, result.job_id, result.error, showJobLog)}
                      {sourceBadge(result.source)}
                    </span>
                  </td>
                  <td title={`${qualityMetricShort(result.quality_metric)} (val)`}>
                    <span className="result-score-summary">
                      <span className="result-score-value">
                        <small>{perClassF1Values(validationPerClassMetrics(result.training_metrics), "pixel").length > 1 ? "F1 сред. (val)" : `${qualityMetricShort(result.quality_metric)} (val)`}</small>
                        <strong className="technical-value">{formatF1Score(result.f1_score)}</strong>
                      </span>
                      <CompactPerClassF1
                        metrics={validationPerClassMetrics(result.training_metrics)}
                        section="pixel"
                      />
                    </span>
                  </td>
                  <td title={qualityMetricShort(result.quality_metric)} data-label={result.test_f1?.f1 == null ? `${qualityMetricShort(result.quality_metric)} (тест)` : undefined}>
                    {result.test_f1?.f1 !== null && result.test_f1?.f1 !== undefined ? (
                      <span className="result-score-summary">
                        <span className="result-score-value">
                          <small>{result.test_f1.aggregation === "macro" ? "F1 сред." : qualityMetricShort(result.quality_metric)}</small>
                          <a className={`badge technical-value test-f1-link ${result.test_f1.status === "current" ? "ok" : result.test_f1.status === "error" ? "error" : "warning"}`}
                            href={`#/test-f1/${result.id}`} title="Посмотреть тестовый F1 на снимках" aria-label={`Посмотреть тестовый F1 сети ${result.model_name}`}>
                            {formatTestF1Percent(result.test_f1.f1)}
                          </a>
                        </span>
                        <CompactPerClassF1
                          metrics={result.test_f1.metrics}
                          section={result.quality_metric === "objects" ? "objects" : "pixel"}
                        />
                      </span>
                    ) : result.test_f1?.status === "queued" || result.test_f1?.status === "running" ? (
                      <span className="badge neutral">расчёт</span>
                    ) : "—"}
                  </td>
                  <td className="technical-value" title="Эпоха лучших весов этого этапа" data-label="Эпоха этапа">{result.epoch ?? "—"}</td>
                  <td className="technical-value" title="Создано" data-label="Создано">{formatTrainingResultDate(result.status, result.trained_at, result.started_at, result.created_at)}</td>
                  <td className="action-cell">
                    {result.status === "ok" ? (
                      <>
                        {result.can_continue_training ? <button className="secondary compact-action" type="button" title="Продолжить обучение от выбранного чекпойнта" onClick={() => onContinue(result)}><Play size={14} />Продолжить обучение</button> : null}
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
                      <td title="ИСТОЧНИК"><span className="source-lines">{imageSourceLabel(item, datasets, imageFolders)}<small className="muted">{item.checkpoint_epoch != null ? `Лучшие веса · эпоха ${item.checkpoint_epoch}${family.stages.length > 1 ? ` этапа ${stageIndex + 1}` : ""}` : "Сохранённые веса сети"}</small></span></td>
                      <td title="Статус">
                        <span className="status-stack">
                          {resultStatusBadge(item.status, "inference", item.progress, item.job_id, undefined, showJobLog)}
                          {sourceBadge(item.source)}
                        </span>
                      </td>
                      <td title="GeoJSON">{item.geojson_file ? <span className="pseudo-download-actions">
                        {geojsonDownloadLink(item.geojson_file)}
                        {item.status === "ok" ? <a className="secondary icon-button" href={`#/pseudo-markup/${item.id}`} title="Посмотреть псевдоразметку на мозаике снимков" aria-label="Посмотреть псевдоразметку на мозаике снимков"><MapIcon size={15} /></a> : null}
                        {item.status === "ok" ? <PseudoCompareButton id={item.id} label={`${payload.class_name || "Класс"} · ${payload.dataset_name} · ${result.model_name} · ${item.source_dataset_name} · ${formatDateTime(item.created_at)}${item.checkpoint_epoch != null ? ` · эпоха ${item.checkpoint_epoch}` : ""}`} /> : null}
                      </span> : "—"}</td>
                      <td title="Создано" data-label="Создано">{pseudoCreatedLabel(item)}</td>
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
  inferenceAvailable,
  run,
  closeModal,
  reload,
}: {
  datasetKey: string;
  result: TrainingResultInfo;
  datasets: DatasetInfo[];
  imageFolders: ImageFolderInfo[];
  inferenceAvailable: boolean;
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
    if (!inferenceAvailable) return;
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
      {!inferenceAvailable ? <div className="inference-template-warning" role="alert">Классу не назначен шаблон инференса. Чтобы создать псевдоразметку, <a href="#/templates/inference" onClick={closeModal}>назначьте шаблон</a> в разделе «Инференс».</div> : null}
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
      <button className="primary" type="submit" disabled={!inferenceAvailable}>
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
      <section
        className={`modal-card${modal.wide ? " wide" : ""}${modal.fullscreen ? " fullscreen" : ""}`}
        role="dialog"
        aria-modal="true"
      >
        <header className="modal-header">
          <h2>{modal.title}</h2>
          <button className="ghost icon-button" type="button" onClick={onClose} aria-label="Закрыть">
            <X size={17} />
          </button>
        </header>
        <div className="modal-body">{modal.body}</div>
        {modal.footer !== null ? <footer className="modal-footer">
          {modal.footer || (
            <button className="secondary" type="button" onClick={onClose}>
              Закрыть
            </button>
          )}
        </footer> : null}
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

function templateFor(templates: TrainingTemplate[], architecture: string, datasetKey: string | null): TrainingTemplate | undefined {
  const datasetTemplate =
    datasetKey && datasetKey !== "custom"
      ? templates.find((item) => item.architecture === architecture && item.dataset_key === datasetKey && item.is_active)
      : undefined;
  return datasetTemplate || templates.find((item) => item.architecture === architecture && !item.dataset_key) || templates[0];
}

function templateTitle(template: TrainingTemplate): string {
  return template.dataset_key ? `${template.display_name} · ${template.dataset_name || template.dataset_key}` : template.display_name;
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
  if (status === "queued" || status === "running" || status === "paused") return status;
  if (status === "error" || status === "failed") return "error";
  if (status === "cancelled") return "warning";
  return "neutral";
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    queued: "в очереди",
    running: "в процессе",
    paused: "приостановлено для более приоритетного задания",
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
  return status === "queued" || status === "running" || status === "paused";
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
      <span>{trainingModelLabel(job.model_name, job.pipeline_variant)}</span>
      {job.tile_size ? <small className="muted">tile={job.tile_size}</small> : null}
    </span>
  );
}

function trainingModelLabel(name: string, variant: string): string {
  return ["next_gen2", "object_f1"].includes(variant) ? `${name.replace(" (next-gen)", "")} (${variant === "object_f1" ? "object f1" : "next-gen2"})` : name;
}

function mergedQueueJobs(snapshot: QueueSnapshot): JobSummary[] {
  return [...(snapshot.training_jobs || []), ...(snapshot.inference_jobs || [])].sort((left, right) => {
    if (left.status !== right.status) return queuePriority(left) - queuePriority(right);
    return left.queue_position - right.queue_position;
  });
}

function queuePriority(job: JobSummary): number {
  if (job.status === "running") return 0;
  if (job.status === "paused") return 1;
  if (job.status === "queued") return 2;
  return 3;
}

type ManagedDatasetDraftSource = {
  dataset: DatasetInfo;
  selected: boolean;
  priority: number;
  color: string;
};

function ManagedDatasetForm({
  targetClass,
  catalog,
  dataset,
  run,
  onCancel,
  onSaved,
}: {
  targetClass: NonNullable<DatasetCatalogInfo["classes"]>[number];
  catalog: DatasetCatalogInfo;
  dataset?: DatasetInfo;
  run: Runner;
  onCancel: () => void;
  onSaved: (catalog: DatasetCatalogInfo) => Promise<void>;
}) {
  const palette = ["#3B82F6", "#22C55E", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4"];
  const candidates = useMemo(
    () => (catalog.classes || []).flatMap((classInfo) => (classInfo.datasets || [])
      .filter((dataset) => (
        !dataset.managed
        && dataset.format === "per_image"
        && dataset.source_available
        && classInfo.imagery_type === targetClass.imagery_type
      ))
      .map((dataset) => ({ dataset, classInfo }))),
    [catalog.classes, targetClass.imagery_type],
  );
  const [name, setName] = useState(dataset?.dataset_name || "main");
  const [sources, setSources] = useState<ManagedDatasetDraftSource[]>(() => {
    const existing = new Map(
      (dataset?.managed_sources || []).map((source) => [source.dataset_key, source]),
    );
    return candidates.map(({ dataset: candidate }, index) => {
      const current = existing.get(candidate.key);
      return {
        dataset: candidate,
        selected: Boolean(current),
        priority: current?.priority ?? Math.max(0, 100 - index * 10),
        color: current?.color || palette[index % palette.length],
      };
    });
  });
  const selected = sources.filter((item) => item.selected);
  const selectedClassKeys = new Set(selected.map((item) => item.dataset.class_key));
  const valid = name.trim().length > 0 && selected.length >= 2
    && selectedClassKeys.size === selected.length;

  const updateSource = (datasetKey: string, patch: Partial<ManagedDatasetDraftSource>) => {
    setSources((current) => current.map((item) => (
      item.dataset.key === datasetKey ? { ...item, ...patch } : item
    )));
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!valid) return;
    const payload = await run(() => apiJson<DatasetCatalogInfo>(
      dataset
        ? `/managed-datasets/${encodeURIComponent(dataset.key)}`
        : "/managed-datasets/compose",
      {
        method: dataset ? "PATCH" : "POST",
        body: {
          ...(dataset ? {} : { class_key: targetClass.key }),
          name: name.trim(),
          sources: selected.map((item) => ({
            dataset_key: item.dataset.key,
            priority: item.priority,
            color: item.color,
          })),
        },
      },
    ));
    if (payload) await onSaved(payload);
  };

  return (
    <form className="form-stack" onSubmit={submit}>
      <p className="muted">
        Разметка строится из выбранных binary per-image датасетов. Большее число означает более высокий
        приоритет в местах наложения.
      </p>
      <label>
        Название датасета
        <input value={name} onChange={(event) => setName(event.target.value)} maxLength={240} required />
      </label>
      <div className="managed-source-picker">
        {sources.map((item) => {
          const sourceClass = candidates.find(({ dataset }) => dataset.key === item.dataset.key)?.classInfo;
          const duplicateClass = !item.selected && selected.some(
            (selectedItem) => selectedItem.dataset.class_key === item.dataset.class_key,
          );
          return (
            <label
              className={`managed-source-row${item.selected ? " selected" : ""}`}
              key={item.dataset.key}
            >
              <input
                type="checkbox"
                checked={item.selected}
                disabled={duplicateClass}
                onChange={(event) => updateSource(item.dataset.key, { selected: event.target.checked })}
              />
              <span className="managed-source-name">
                <strong>{sourceClass?.name}\\{item.dataset.dataset_name}</strong>
                <small>{item.dataset.image_count ?? 0} снимков</small>
              </span>
              <span>
                Приоритет
                <input
                  type="number"
                  value={item.priority}
                  disabled={!item.selected}
                  onChange={(event) => updateSource(item.dataset.key, {
                    priority: Number.parseInt(event.target.value || "0", 10),
                  })}
                />
              </span>
              <span>
                Цвет
                <input
                  type="color"
                  value={item.color}
                  disabled={!item.selected}
                  onChange={(event) => updateSource(item.dataset.key, { color: event.target.value })}
                />
              </span>
            </label>
          );
        })}
      </div>
      {!candidates.length ? (
        <div className="notice warning">Нет доступных binary per-image датасетов с тем же типом снимков.</div>
      ) : null}
      {selected.length >= 2 && selectedClassKeys.size !== selected.length ? (
        <div className="notice warning">Выберите не более одного датасета каждого исходного класса.</div>
      ) : null}
      <div className="button-row modal-form-actions">
        <button className="secondary" type="button" onClick={onCancel}>Отмена</button>
        <button className="primary" type="submit" disabled={!valid}>
          {dataset ? "Сохранить" : "Создать"}
        </button>
      </div>
    </form>
  );
}

type ClassF1Value = {
  slug: string;
  name: string;
  color: string;
  f1: number | null;
};

const CLASS_F1_COLLATOR = new Intl.Collator("ru", {
  sensitivity: "base",
  numeric: true,
});

function compareClassF1Values(left: ClassF1Value, right: ClassF1Value): number {
  return CLASS_F1_COLLATOR.compare(left.name, right.name)
    || CLASS_F1_COLLATOR.compare(left.slug, right.slug);
}

export function perClassF1Values(metrics: unknown, section: "pixel" | "objects"): ClassF1Value[] {
  if (!metrics || typeof metrics !== "object") return [];
  const sectionValue = (metrics as Record<string, unknown>)[section];
  if (!sectionValue || typeof sectionValue !== "object") return [];
  const perClass = (sectionValue as Record<string, unknown>).per_class;
  if (!perClass || typeof perClass !== "object") return [];
  return Object.entries(perClass as Record<string, unknown>).flatMap(([slug, raw]) => {
    if (!raw || typeof raw !== "object") return [];
    const item = raw as Record<string, unknown>;
    return [{
      slug: String(item.slug || slug),
      name: String(item.name || item.slug || slug),
      color: /^#[0-9A-Fa-f]{6}$/.test(String(item.color || "")) ? String(item.color) : "#808080",
      f1: typeof item.f1 === "number" ? item.f1 : null,
    }];
  }).sort(compareClassF1Values);
}

function metricAggregationLabel(
  metrics: unknown,
  qualityMetric: "pixel" | "objects",
): string {
  const aggregation = metrics && typeof metrics === "object"
    ? (metrics as Record<string, unknown>).aggregation
    : null;
  return aggregation === "macro" ? "F1 сред." : qualityMetricShort(qualityMetric);
}

function PerClassF1Table({ metrics }: { metrics: unknown }) {
  const pixel = perClassF1Values(metrics, "pixel");
  const objects = perClassF1Values(metrics, "objects");
  const pixelBySlug = new Map(pixel.map((item) => [item.slug, item]));
  const objectsBySlug = new Map(objects.map((item) => [item.slug, item]));
  const rows = [...new Set([...pixel.map((item) => item.slug), ...objects.map((item) => item.slug)])]
    .map((slug) => pixelBySlug.get(slug) || objectsBySlug.get(slug)!)
    .sort(compareClassF1Values);
  if (!rows.length) return null;
  return (
    <div className="table-wrap multiclass-f1-table-wrap">
      <table className="multiclass-f1-table">
        <thead><tr><th>Тип объекта</th><th>F1 пиксельный</th><th>F1 объектовый</th></tr></thead>
        <tbody>
          {rows.map((item) => {
            const { slug } = item;
            return (
              <tr key={slug}>
                <td><span className="inline-row"><span className="class-color-dot" style={{ backgroundColor: item.color }} />{item.name}<small className="technical-value muted">{slug}</small></span></td>
                <td className="technical-value">{formatF1Score(pixelBySlug.get(slug)?.f1)}</td>
                <td className="technical-value">{formatF1Score(objectsBySlug.get(slug)?.f1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CompactPerClassF1({ metrics, section }: { metrics: unknown; section: "pixel" | "objects" }) {
  const values = perClassF1Values(metrics, section);
  if (!values.length) return null;
  return (
    <span className="compact-class-f1-list">
      {values.map((item) => (
        <small className="compact-class-f1-chip" key={item.slug} title={`${item.name} · ${item.slug}`}>
          <span className="class-color-dot" style={{ backgroundColor: item.color }} />
          <span className="compact-class-f1-name">{item.name}</span>
          <strong>{item.f1 === null ? "—" : formatTestF1Percent(item.f1)}</strong>
        </small>
      ))}
    </span>
  );
}

function validationPerClassMetrics(trainingMetrics: unknown): Record<string, unknown> {
  if (!trainingMetrics || typeof trainingMetrics !== "object") return {};
  const raw = (trainingMetrics as Record<string, unknown>).val_per_class_metrics;
  if (!Array.isArray(raw)) return {};
  const perClass = Object.fromEntries(raw.flatMap((value) => {
    if (!value || typeof value !== "object") return [];
    const item = value as Record<string, unknown>;
    const slug = String(item.slug || "");
    return slug ? [[slug, item]] : [];
  }));
  return { pixel: { per_class: perClass } };
}

function automationRuleKey(datasetKey: string, architecture: string): string {
  return `${datasetKey}::${architecture}`;
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
