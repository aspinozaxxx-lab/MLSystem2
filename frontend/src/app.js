const API = "/api/v1";
const root = document.getElementById("app");
const PROGRESS_REFRESH_MS = 10000;
let progressRefreshTimer = null;

const state = {
  user: null,
  links: [],
  datasets: [],
  imageFolders: [],
  classes: [],
  models: [],
  templates: [],
  inferenceTemplates: [],
  experiments: [],
  selectedArchitecture: "",
  selectedTemplateId: "",
  selectedInferenceTemplateId: "",
  modal: null,
};

window.addEventListener("hashchange", () => {
  clearProgressRefreshTimer();
  render();
});
init();

async function init() {
  const me = await apiJson("/auth/me", { authOptional: true });
  state.user = me && me.authenticated ? me.username : null;
  await render();
}

async function render() {
  clearProgressRefreshTimer();
  if (!state.user) {
    renderLogin();
    return;
  }
  await ensureSharedData();
  const route = currentRoute();
  if (route[0] === "start") return renderStartPage();
  if (route[0] === "queue") return renderQueuePage();
  if (route[0] === "templates") return renderTemplatesPage();
  if (route[0] === "automation") return renderAutomationPage();
  if (route[0] === "model-export") return renderModelExportPage();
  if (route[0] === "results" && route[1]) return renderClassResultPage(decodeURIComponent(route[1]));
  if (route[0] === "results") return renderResultsPage();
  if (route[0] === "jobs" && route[1]) return renderJobPage(route[1]);
  return renderHomePage();
}

function currentRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return hash ? hash.split("/") : [];
}

function navigate(path) {
  window.location.hash = path;
}

function clearProgressRefreshTimer() {
  if (progressRefreshTimer !== null) {
    window.clearTimeout(progressRefreshTimer);
    progressRefreshTimer = null;
  }
}

function scheduleProgressRefresh(callback, enabled) {
  clearProgressRefreshTimer();
  if (!enabled) return;
  progressRefreshTimer = window.setTimeout(() => {
    progressRefreshTimer = null;
    if (state.modal) {
      scheduleProgressRefresh(callback, true);
      return;
    }
    callback();
  }, PROGRESS_REFRESH_MS);
}

async function ensureSharedData() {
  const [links, datasets, imageFolders, classes, models, templates, inferenceTemplates] = await Promise.all([
    apiJson("/app-links"),
    apiJson("/datasets"),
    apiJson("/image-folders"),
    apiJson("/classes"),
    apiJson("/models"),
    apiJson("/training-templates"),
    apiJson("/inference-templates"),
  ]);
  state.links = links.links || [];
  state.datasets = datasets.datasets || [];
  state.imageFolders = imageFolders.folders || [];
  state.classes = classes.classes || [];
  state.models = models.models || [];
  state.templates = templates.templates || [];
  state.inferenceTemplates = inferenceTemplates.templates || [];
  if (!state.selectedArchitecture && state.models.length) {
    state.selectedArchitecture = state.models[0].architecture;
  }
  if (state.selectedTemplateId && !state.templates.some((item) => item.id === state.selectedTemplateId)) {
    state.selectedTemplateId = "";
  }
  if (!state.selectedTemplateId && state.templates.length) {
    state.selectedTemplateId = selectedTemplate()?.id || state.templates[0].id;
  }
  if (
    state.selectedInferenceTemplateId
    && !state.inferenceTemplates.some((item) => item.id === state.selectedInferenceTemplateId)
  ) {
    state.selectedInferenceTemplateId = "";
  }
  if (!state.selectedInferenceTemplateId && state.inferenceTemplates.length) {
    state.selectedInferenceTemplateId = selectedInferenceTemplate()?.id || state.inferenceTemplates[0].id;
  }
}

async function loadExperimentsSafe() {
  const response = await fetch(`${API}/mlflow/experiments`, { credentials: "same-origin" });
  if (response.status === 401) {
    state.user = null;
    renderLogin();
    return [];
  }
  if (!response.ok) {
    return [];
  }
  return response.json();
}

function renderShell(content) {
  document.body.classList.remove("login-body");
  root.innerHTML = `
    <header class="topbar">
      <a class="brand" href="#/">MLSystem2</a>
      <nav class="nav">
        <a href="#/start">Запуск обучения</a>
        <a href="#/queue">В процессе</a>
        <a href="#/templates">Шаблоны</a>
        <a href="#/automation">Автоматизация</a>
        <a href="#/model-export">Экспорт модели</a>
        <a href="#/results">Результаты</a>
        <button class="link-button" id="logout-button" type="button">Выйти</button>
      </nav>
    </header>
    <main class="page">${content}</main>
    <div id="modal-root">${state.modal || ""}</div>
  `;
  document.getElementById("logout-button").addEventListener("click", logout);
}

function renderLogin(error = "") {
  document.body.classList.add("login-body");
  root.innerHTML = `
    <main class="login-card">
      <div class="login-mark">MLSystem2</div>
      <h1>Вход</h1>
      <form id="login-form" class="form-stack">
        <label>Логин
          <input name="username" autocomplete="username" required>
        </label>
        <label>Пароль
          <input name="password" type="password" autocomplete="current-password" required>
        </label>
        ${error ? `<div class="error">${escapeHtml(error)}</div>` : ""}
        <button type="submit" class="primary">Войти</button>
      </form>
    </main>
  `;
  document.getElementById("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const response = await fetch(`${API}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({
        username: String(form.get("username") || ""),
        password: String(form.get("password") || ""),
      }),
    });
    if (!response.ok) {
      renderLogin("Неверный логин или пароль");
      return;
    }
    state.user = String(form.get("username") || "");
    await render();
  });
}

async function logout() {
  await fetch(`${API}/auth/logout`, { method: "POST", credentials: "same-origin" });
  state.user = null;
  renderLogin();
}

async function renderHomePage() {
  const linkMap = Object.fromEntries(state.links.map((item) => [item.key, item.url]));
  const changes = await apiJson("/results/changes");
  renderShell(`
    <section class="hero">
      <h1>MLSystem2</h1>
      <p>Панель запуска обучения, очередей и результатов моделей</p>
    </section>
    <section class="grid">
      <a class="card active" href="${escapeAttr(linkMap.grafana || "#")}" target="_blank" rel="noreferrer">
        <div>
          <h2>Grafana</h2>
          <p>Мониторинг ресурсов и состояния сервисов.</p>
        </div>
        <span class="secondary">Открыть</span>
      </a>
      <a class="card active" href="${escapeAttr(linkMap.mlflow || "#")}" target="_blank" rel="noreferrer">
        <div>
          <h2>MLflow</h2>
          <p>Эксперименты, runs, метрики и артефакты.</p>
        </div>
        <span class="secondary">Открыть</span>
      </a>
      <a class="card active" href="${escapeAttr(linkMap.minio || "#")}" target="_blank" rel="noreferrer">
        <div>
          <h2>MinIO</h2>
          <p>Снимки и артефакты в объектном хранилище.</p>
        </div>
        <span class="secondary">Открыть</span>
      </a>
      <a class="card active" href="${escapeAttr(linkMap.open_webui || "#")}" target="_blank" rel="noreferrer">
        <div>
          <h2>Open WebUI</h2>
          <p>Веб-интерфейс LLM-стека Ollama.</p>
        </div>
        <span class="secondary">Открыть</span>
      </a>
    </section>

    <h2 class="section-title">Обучение моделей</h2>
    <section class="grid">
      ${homeTrainingCard("Запуск обучения", "Форма создания training job.", "#/start")}
      ${homeTrainingCard("В процессе", "Очереди обучения и инференса.", "#/queue")}
      ${homeTrainingCard("Шаблоны", "Параметры по умолчанию для обучения и инференса.", "#/templates")}
      ${homeTrainingCard("Автоматизация", "Автозапуск обучения и псевдоразметки по версиям датасетов.", "#/automation")}
      ${homeTrainingCard("Экспорт модели", "Архив для регистрации модели в Triton CPU service.", "#/model-export")}
      ${homeTrainingCard("Результаты", "Классы, метрики и псевдоразметка.", "#/results")}
    </section>
    <section class="panel">
      <h2>Последние изменения</h2>
      ${renderResultChangesTable(changes.changes || [], { activeClick: "class" })}
    </section>
  `);
  bindResultChangeRows({ activeClick: "class" });
}

function homeTrainingCard(title, text, href) {
  return `
    <a class="card active group" href="${href}">
      <div>
        <h2>${title}</h2>
        <p>${text}</p>
      </div>
      <span class="secondary">Перейти</span>
    </a>
  `;
}

async function renderStartPage() {
  state.experiments = await loadExperimentsSafe();
  const initialDatasetKey = state.datasets[0]?.key || "";
  const template = templateFor(state.selectedArchitecture, initialDatasetKey);
  const datasets = state.datasets.map((item) => option(item.key, item.name, false)).join("");
  const models = state.models.map((item) => option(item.architecture, item.display_name, item.architecture === state.selectedArchitecture)).join("");
  const defaultExperimentId = latestExperimentId();
  const experimentsOptions = [
    `<option value="" ${defaultExperimentId ? "" : "selected"}>Новый experiment</option>`,
    ...state.experiments.map((item) => option(item.experiment_id, item.name, item.experiment_id === defaultExperimentId)),
  ].join("");
  renderShell(`
    <section class="hero compact-hero">
      <h1>Запуск обучения</h1>
      <p>Создание training job в очереди MLSystem2</p>
    </section>
    <form id="start-form" class="form-stack">
      <section class="panel">
        <div class="form-grid">
          <label>MLflow experiment
            <select name="experiment_id" id="experiment-select">${experimentsOptions}</select>
          </label>
          <label id="new-experiment-name-field" class="${defaultExperimentId ? "hidden" : ""}">Новое имя experiment
            <input name="experiment_name" value="MLSystem2">
          </label>
          <label>Датасет
            <select name="dataset_key" id="dataset-select">${datasets}</select>
          </label>
          <label>Модель
            <select name="architecture" id="architecture-select">${models}</select>
          </label>
        </div>
      </section>
      <section class="panel hidden" id="custom-upload">
        <h2>Custom dataset</h2>
        <div class="form-grid">
          <label>GeoJSON
            <input name="annotation_geojson" type="file" accept=".geojson,application/geo+json">
          </label>
          <label>TXT со снимками
            <input name="scenes_txt" type="file" accept=".txt,text/plain">
          </label>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header">
          <h2>Параметры</h2>
          <span class="badge neutral" id="selected-template-badge">${escapeHtml(template.display_name)}</span>
        </div>
        <div class="form-grid" id="config-fields">
          ${renderConfigFields(template.config_schema, template.default_config)}
        </div>
      </section>
      <button class="primary" type="submit">Запустить обучение</button>
    </form>
  `);

  const datasetSelect = document.getElementById("dataset-select");
  const experimentSelect = document.getElementById("experiment-select");
  const newExperimentNameField = document.getElementById("new-experiment-name-field");
  const uploadPanel = document.getElementById("custom-upload");
  const syncTemplateFields = () => {
    const currentTemplate = templateFor(state.selectedArchitecture, datasetSelect.value);
    document.getElementById("selected-template-badge").textContent = currentTemplate.display_name;
    document.getElementById("config-fields").innerHTML = renderConfigFields(
      currentTemplate.config_schema,
      currentTemplate.default_config
    );
  };
  const syncUpload = () => uploadPanel.classList.toggle("hidden", datasetSelect.value !== "custom");
  const syncExperimentName = () => {
    newExperimentNameField.classList.toggle("hidden", experimentSelect.value !== "");
  };
  experimentSelect.addEventListener("change", syncExperimentName);
  datasetSelect.addEventListener("change", () => {
    syncUpload();
    syncTemplateFields();
  });
  syncUpload();
  syncExperimentName();

  document.getElementById("architecture-select").addEventListener("change", async (event) => {
    state.selectedArchitecture = event.currentTarget.value;
    await renderStartPage();
  });
  document.getElementById("start-form").addEventListener("submit", submitStartForm);
}

function renderModelExportPage() {
  renderShell(`
    <section class="hero compact-hero">
      <h1>Экспорт модели</h1>
      <p>Сборка zip-архива для models-serving-service и Triton CPU</p>
    </section>
    <form id="model-export-form" class="form-stack">
      <section class="panel">
        <div class="form-grid">
          <label>Имя модели
            <input
              name="model_name"
              required
              pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
              placeholder="deforestation-b2"
              autocomplete="off"
            >
          </label>
          <label>Checkpoint .pt
            <input name="checkpoint" type="file" accept=".pt" required>
          </label>
        </div>
      </section>
      <div class="inline-row">
        <button class="primary" type="submit" id="model-export-submit">Собрать zip</button>
        <span class="info-box hidden" id="model-export-status"></span>
      </div>
    </form>
  `);
  document.getElementById("model-export-form").addEventListener("submit", submitModelExportForm);
}

async function submitModelExportForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = document.getElementById("model-export-submit");
  const status = document.getElementById("model-export-status");
  const data = new FormData(form);
  const modelName = String(data.get("model_name") || "").trim();
  const checkpoint = data.get("checkpoint");
  if (!isValidExportModelName(modelName)) {
    showModal("Ошибка", "Имя модели должно содержать только a-z, 0-9, дефис и подчеркивание.", "Понятно");
    return;
  }
  if (!(checkpoint instanceof File) || !checkpoint.name || !checkpoint.name.toLowerCase().endsWith(".pt")) {
    showModal("Ошибка", "Выберите MLSystem2 checkpoint .pt.", "Понятно");
    return;
  }
  await exportModelArchive(modelName, checkpoint, null, button, status);
}

async function exportModelArchive(modelName, checkpoint, sampleSize, button, status) {
  const request = new FormData();
  request.set("model_name", modelName);
  request.set("checkpoint", checkpoint);
  if (sampleSize !== null && sampleSize !== undefined) request.set("sample_size", String(sampleSize));

  button.disabled = true;
  status.textContent = "Идет экспорт...";
  status.classList.remove("hidden");
  try {
    const response = await fetch(`${API}/model-export/triton-zip`, {
      method: "POST",
      credentials: "same-origin",
      body: request,
    });
    if (response.status === 401) {
      state.user = null;
      renderLogin();
      throw new Error("auth");
    }
    if (!response.ok) {
      const message = await errorMessage(response);
      if (message.includes("metadata.sample_size")) {
        status.textContent = "Нужен sample_size.";
        showModelExportSampleSizeModal(modelName, checkpoint, button, status);
        return;
      }
      showModal("Ошибка", message, "Понятно");
      throw new Error(message);
    }
    const blob = await response.blob();
    downloadBlob(blob, downloadFilename(response) || `${modelName}_export.zip`);
    status.textContent = "Архив скачан.";
  } finally {
    button.disabled = false;
  }
}

function showModelExportSampleSizeModal(modelName, checkpoint, button, status) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Укажите sample_size</h2>
        <p>В checkpoint нет metadata.sample_size. Это старый checkpoint, поэтому размер тайла надо задать вручную.</p>
        <form id="model-export-sample-size-form" class="form-stack">
          <label>sample_size
            <input name="sample_size" type="number" min="32" step="32" value="768" required>
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Продолжить экспорт</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("model-export-sample-size-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const sampleSize = Number.parseInt(String(data.get("sample_size") || ""), 10);
    if (!Number.isInteger(sampleSize) || sampleSize <= 0 || sampleSize % 32 !== 0) {
      showModal("Ошибка", "sample_size должен быть положительным числом, кратным 32.", "Понятно");
      return;
    }
    closeModal();
    await exportModelArchive(modelName, checkpoint, sampleSize, button, status);
  });
}

function showTrainingResultZipModal(resultId, defaultModelName) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Скачать zip</h2>
        <form id="result-zip-form" class="form-stack">
          <label>Имя сети
            <input
              name="model_name"
              required
              pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
              value="${escapeAttr(defaultModelName)}"
              autocomplete="off"
            >
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Скачать zip</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("result-zip-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const modelName = String(data.get("model_name") || "").trim();
    if (!isValidExportModelName(modelName)) {
      showModal("Ошибка", "Имя модели должно содержать только a-z, 0-9, дефис и подчеркивание.", "Понятно");
      return;
    }
    closeModal();
    await exportTrainingResultArchive(resultId, modelName, null);
  });
}

async function exportTrainingResultArchive(resultId, modelName, sampleSize) {
  const request = new FormData();
  request.set("model_name", modelName);
  if (sampleSize !== null && sampleSize !== undefined) request.set("sample_size", String(sampleSize));

  const response = await fetch(`${API}/results/training/${encodeURIComponent(resultId)}/triton-zip`, {
    method: "POST",
    credentials: "same-origin",
    body: request,
  });
  if (response.status === 401) {
    state.user = null;
    renderLogin();
    throw new Error("auth");
  }
  if (!response.ok) {
    const message = await errorMessage(response);
    if (message.includes("metadata.sample_size")) {
      showTrainingResultZipSampleSizeModal(resultId, modelName);
      return;
    }
    showModal("Ошибка", message, "Понятно");
    throw new Error(message);
  }
  const blob = await response.blob();
  downloadBlob(blob, downloadFilename(response) || `${modelName}_export.zip`);
}

function showTrainingResultZipSampleSizeModal(resultId, modelName) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Укажите sample_size</h2>
        <p>В checkpoint нет metadata.sample_size. Это старый checkpoint, поэтому размер тайла надо задать вручную.</p>
        <form id="result-zip-sample-size-form" class="form-stack">
          <label>sample_size
            <input name="sample_size" type="number" min="32" step="32" value="768" required>
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Продолжить экспорт</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("result-zip-sample-size-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const sampleSize = Number.parseInt(String(data.get("sample_size") || ""), 10);
    if (!Number.isInteger(sampleSize) || sampleSize <= 0 || sampleSize % 32 !== 0) {
      showModal("Ошибка", "sample_size должен быть положительным числом, кратным 32.", "Понятно");
      return;
    }
    closeModal();
    await exportTrainingResultArchive(resultId, modelName, sampleSize);
  });
}

async function submitStartForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  let customDatasetId = null;
  const datasetKey = String(data.get("dataset_key") || "");
  if (datasetKey === "custom") {
    const customForm = new FormData();
    customForm.set("name", "Custom");
    const scenesFile = data.get("scenes_txt");
    const geojsonFile = data.get("annotation_geojson");
    if (!(scenesFile instanceof File) || !scenesFile.name || !(geojsonFile instanceof File) || !geojsonFile.name) {
      showModal("Ошибка", "Для Custom нужны geojson и txt.", "Понятно");
      return;
    }
    customForm.set("scenes_txt", scenesFile);
    customForm.set("annotation_geojson", geojsonFile);
    const custom = await apiForm("/custom-datasets", customForm);
    customDatasetId = custom.id;
  }
  let experimentId = String(data.get("experiment_id") || "") || null;
  let experimentName = String(data.get("experiment_name") || "MLSystem2").trim();
  if (experimentId) {
    const select = form.querySelector("select[name='experiment_id']");
    experimentName = select.options[select.selectedIndex]?.textContent || experimentName;
  } else {
    const experiment = await apiJson("/mlflow/experiments", {
      method: "POST",
      body: { name: experimentName },
    });
    experimentId = experiment.experiment_id;
    experimentName = experiment.name;
  }
  const payload = {
    mlflow_experiment_id: experimentId,
    mlflow_experiment_name: experimentName,
    dataset_key: datasetKey,
    custom_dataset_id: customDatasetId,
    architecture: String(data.get("architecture") || state.selectedArchitecture),
    config: collectConfig(form),
  };
  await apiJson("/training-jobs", { method: "POST", body: payload });
  showModal("Обучение запущено", "Задание добавлено в очередь обучения.", "Закрыть", async () => {
    await ensureSharedData();
    renderStartPage();
  });
}

function renderTemplatesPage() {
  const template = selectedTemplate();
  const inferenceTemplate = selectedInferenceTemplate();
  const deleteButton = template.dataset_key
    ? `<button class="danger" type="button" id="template-delete">Удалить</button>`
    : "";
  const inferenceDeleteButton = inferenceTemplate.dataset_key
    ? `<button class="danger" type="button" id="inference-template-delete">Удалить</button>`
    : "";
  renderShell(`
    <section class="hero compact-hero">
      <h1>Шаблоны</h1>
      <p>Базовые defaults сети и переопределения для конкретных датасетов</p>
    </section>
    <section class="templates-layout">
      <aside class="panel template-tree-panel">
        <div class="panel-header">
          <h2>Шаблоны обучения</h2>
          <button class="secondary compact-action" type="button" id="template-add">Добавить шаблон</button>
        </div>
        <div class="template-tree">${renderTemplateTree()}</div>
      </aside>
      <section class="panel template-editor-panel">
        <div class="panel-header">
          <div>
            <h2>${escapeHtml(templateTitle(template))}</h2>
            <p class="muted">${template.dataset_key ? "Шаблон датасета" : "Базовый шаблон сети"}</p>
          </div>
          <div class="inline-row template-meta">
            <span class="badge ${template.source === "manual" ? "warning" : "ok"}">source=${template.source}</span>
            <span class="badge neutral">version=${template.version}</span>
          </div>
        </div>
        <form id="template-form" class="form-stack">
          <div class="form-grid">${renderConfigFields(template.config_schema, template.default_config, false, { applyAll: true })}</div>
          <div class="inline-row">
            <button class="primary" type="submit">Сохранить</button>
            <button class="secondary" type="button" id="template-reset">Сбросить</button>
            ${deleteButton}
          </div>
        </form>
      </section>
    </section>
    <section class="templates-layout">
      <aside class="panel template-tree-panel">
        <div class="panel-header">
          <h2>Шаблоны инференса</h2>
          <button class="secondary compact-action" type="button" id="inference-template-add">Добавить шаблон</button>
        </div>
        <div class="template-tree">${renderInferenceTemplateTree()}</div>
      </aside>
      <section class="panel template-editor-panel">
        <div class="panel-header">
          <div>
            <h2>${escapeHtml(templateTitle(inferenceTemplate))}</h2>
            <p class="muted">${inferenceTemplate.dataset_key ? "Шаблон датасета" : "Базовый шаблон сети"}</p>
          </div>
          <div class="inline-row template-meta">
            <span class="badge ${inferenceTemplate.source === "manual" ? "warning" : "ok"}">source=${inferenceTemplate.source}</span>
            <span class="badge neutral">version=${inferenceTemplate.version}</span>
          </div>
        </div>
        <form id="inference-template-form" class="form-stack">
          <div class="form-grid">${renderConfigFields(inferenceTemplate.config_schema, inferenceTemplate.default_config, false, { applyAll: true })}</div>
          <div class="inline-row">
            <button class="primary" type="submit">Сохранить</button>
            <button class="secondary" type="button" id="inference-template-reset">Сбросить</button>
            ${inferenceDeleteButton}
          </div>
        </form>
      </section>
    </section>
  `);
  for (const button of document.querySelectorAll(".template-tree-button")) {
    button.addEventListener("click", () => {
      state.selectedTemplateId = button.dataset.templateId;
      state.selectedArchitecture = button.dataset.architecture;
      renderTemplatesPage();
    });
  }
  document.getElementById("template-add").addEventListener("click", showTemplateCreateModal);
  document.getElementById("template-form").addEventListener("click", async (event) => {
    const button = event.target.closest(".field-apply-all");
    if (!button) return;
    event.preventDefault();
    const config = collectConfig(document.getElementById("template-form"));
    await apiJson(`/training-templates/by-id/${encodeURIComponent(template.id)}/apply-field-to-all`, {
      method: "PUT",
      body: { key: button.dataset.applyKey, value: config[button.dataset.applyKey] },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  document.getElementById("template-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await apiJson(`/training-templates/by-id/${encodeURIComponent(template.id)}`, {
      method: "PUT",
      body: { default_config: collectConfig(event.currentTarget) },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  document.getElementById("template-reset").addEventListener("click", async () => {
    await apiJson(`/training-templates/by-id/${encodeURIComponent(template.id)}`, {
      method: "PUT",
      body: { reset_to_baseline: true },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  if (template.dataset_key) {
    document.getElementById("template-delete").addEventListener("click", () => showTemplateDeleteModal(template));
  }
  for (const button of document.querySelectorAll(".inference-template-tree-button")) {
    button.addEventListener("click", () => {
      state.selectedInferenceTemplateId = button.dataset.templateId;
      state.selectedArchitecture = button.dataset.architecture;
      renderTemplatesPage();
    });
  }
  document.getElementById("inference-template-add").addEventListener("click", showInferenceTemplateCreateModal);
  document.getElementById("inference-template-form").addEventListener("click", async (event) => {
    const button = event.target.closest(".field-apply-all");
    if (!button) return;
    event.preventDefault();
    const config = collectConfig(document.getElementById("inference-template-form"));
    await apiJson(`/inference-templates/by-id/${encodeURIComponent(inferenceTemplate.id)}/apply-field-to-all`, {
      method: "PUT",
      body: { key: button.dataset.applyKey, value: config[button.dataset.applyKey] },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  document.getElementById("inference-template-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await apiJson(`/inference-templates/by-id/${encodeURIComponent(inferenceTemplate.id)}`, {
      method: "PUT",
      body: { default_config: collectConfig(event.currentTarget) },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  document.getElementById("inference-template-reset").addEventListener("click", async () => {
    await apiJson(`/inference-templates/by-id/${encodeURIComponent(inferenceTemplate.id)}`, {
      method: "PUT",
      body: { reset_to_baseline: true },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  if (inferenceTemplate.dataset_key) {
    document
      .getElementById("inference-template-delete")
      .addEventListener("click", () => showInferenceTemplateDeleteModal(inferenceTemplate));
  }
}

function renderTemplateTree() {
  return baseTemplates().map((parent) => {
    const children = datasetTemplates(parent.architecture);
    return `
      <div class="template-tree-group">
        ${renderTemplateTreeButton(parent, "parent")}
        <div class="template-tree-children">
          ${children.length
            ? children.map((child) => renderTemplateTreeButton(child, "child")).join("")
            : `<span class="template-empty-child">датасетных шаблонов нет</span>`}
        </div>
      </div>
    `;
  }).join("");
}

function renderTemplateTreeButton(template, level) {
  const active = template.id === selectedTemplate().id ? "active" : "";
  return `
    <button
      class="template-tree-button ${level} ${active}"
      type="button"
      data-template-id="${escapeAttr(template.id)}"
      data-architecture="${escapeAttr(template.architecture)}"
    >
      <span>${escapeHtml(templateTitle(template))}</span>
      <small>${template.dataset_key ? "датасет" : "сеть"}</small>
    </button>
  `;
}

function renderInferenceTemplateTree() {
  return baseInferenceTemplates().map((parent) => {
    const children = datasetInferenceTemplates(parent.architecture);
    return `
      <div class="template-tree-group">
        ${renderInferenceTemplateTreeButton(parent, "parent")}
        <div class="template-tree-children">
          ${children.length
            ? children.map((child) => renderInferenceTemplateTreeButton(child, "child")).join("")
            : `<span class="template-empty-child">датасетных шаблонов нет</span>`}
        </div>
      </div>
    `;
  }).join("");
}

function renderInferenceTemplateTreeButton(template, level) {
  const selected = selectedInferenceTemplate();
  const active = selected && template.id === selected.id ? "active" : "";
  return `
    <button
      class="template-tree-button inference-template-tree-button ${level} ${active}"
      type="button"
      data-template-id="${escapeAttr(template.id)}"
      data-architecture="${escapeAttr(template.architecture)}"
    >
      <span>${escapeHtml(templateTitle(template))}</span>
      <small>${template.dataset_key ? "датасет" : "сеть"}</small>
    </button>
  `;
}

function showTemplateCreateModal() {
  const selected = selectedTemplate();
  const parentOptions = baseTemplates()
    .map((item) => option(item.architecture, item.display_name, item.architecture === selected.architecture))
    .join("");
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Добавить шаблон</h2>
        <form id="template-create-form" class="form-stack">
          <label>Архитектура
            <select name="architecture" id="template-create-architecture">${parentOptions}</select>
          </label>
          <label>Датасет
            <select name="dataset_key" id="template-create-dataset"></select>
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Создать</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  const architectureSelect = document.getElementById("template-create-architecture");
  const datasetSelect = document.getElementById("template-create-dataset");
  const syncDatasets = () => {
    datasetSelect.innerHTML = availableDatasetTemplateOptions(architectureSelect.value);
  };
  architectureSelect.addEventListener("change", syncDatasets);
  syncDatasets();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("template-create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const created = await apiJson("/training-templates", {
      method: "POST",
      body: {
        architecture: String(form.get("architecture") || ""),
        dataset_key: String(form.get("dataset_key") || ""),
      },
    });
    state.selectedTemplateId = created.id;
    state.selectedArchitecture = created.architecture;
    closeModal();
    await ensureSharedData();
    renderTemplatesPage();
  });
}

function showInferenceTemplateCreateModal() {
  const selected = selectedInferenceTemplate();
  const parentOptions = baseInferenceTemplates()
    .map((item) => option(item.architecture, item.display_name, item.architecture === selected.architecture))
    .join("");
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Добавить шаблон инференса</h2>
        <form id="inference-template-create-form" class="form-stack">
          <label>Архитектура
            <select name="architecture" id="inference-template-create-architecture">${parentOptions}</select>
          </label>
          <label>Датасет
            <select name="dataset_key" id="inference-template-create-dataset"></select>
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Создать</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  const architectureSelect = document.getElementById("inference-template-create-architecture");
  const datasetSelect = document.getElementById("inference-template-create-dataset");
  const syncDatasets = () => {
    datasetSelect.innerHTML = availableInferenceDatasetTemplateOptions(architectureSelect.value);
  };
  architectureSelect.addEventListener("change", syncDatasets);
  syncDatasets();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("inference-template-create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const created = await apiJson("/inference-templates", {
      method: "POST",
      body: {
        architecture: String(form.get("architecture") || ""),
        dataset_key: String(form.get("dataset_key") || ""),
      },
    });
    state.selectedInferenceTemplateId = created.id;
    state.selectedArchitecture = created.architecture;
    closeModal();
    await ensureSharedData();
    renderTemplatesPage();
  });
}

function showTemplateDeleteModal(template) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Удалить шаблон?</h2>
        <p>Будет удален только шаблон датасета «${escapeHtml(template.dataset_name || template.display_name)}». Базовый шаблон сети сохранится и снова станет fallback.</p>
        <div class="inline-row">
          <button class="danger" type="button" id="template-delete-confirm">Удалить</button>
          <button class="secondary" type="button" id="modal-close">Отмена</button>
        </div>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("template-delete-confirm").addEventListener("click", async () => {
    await apiJson(`/training-templates/by-id/${encodeURIComponent(template.id)}`, { method: "DELETE" });
    const parent = baseTemplates().find((item) => item.architecture === template.architecture) || state.templates[0];
    state.selectedTemplateId = parent?.id || "";
    state.selectedArchitecture = parent?.architecture || state.selectedArchitecture;
    closeModal();
    await ensureSharedData();
    renderTemplatesPage();
  });
}

function showInferenceTemplateDeleteModal(template) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Удалить шаблон инференса?</h2>
        <p>Будет удалён только шаблон датасета «${escapeHtml(template.dataset_name || template.display_name)}». Базовый шаблон сети сохранится и снова станет fallback.</p>
        <div class="inline-row">
          <button class="danger" type="button" id="inference-template-delete-confirm">Удалить</button>
          <button class="secondary" type="button" id="modal-close">Отмена</button>
        </div>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("inference-template-delete-confirm").addEventListener("click", async () => {
    await apiJson(`/inference-templates/by-id/${encodeURIComponent(template.id)}`, { method: "DELETE" });
    const parent = baseInferenceTemplates().find((item) => item.architecture === template.architecture)
      || state.inferenceTemplates[0];
    state.selectedInferenceTemplateId = parent?.id || "";
    state.selectedArchitecture = parent?.architecture || state.selectedArchitecture;
    closeModal();
    await ensureSharedData();
    renderTemplatesPage();
  });
}

function availableDatasetTemplateOptions(architecture) {
  const existing = new Set(datasetTemplates(architecture).map((item) => item.dataset_key));
  const options = state.datasets
    .filter((item) => item.key !== "custom" && !existing.has(item.key))
    .map((item) => option(item.key, item.name, false));
  return options.length ? options.join("") : `<option value="">Нет доступных датасетов</option>`;
}

function availableInferenceDatasetTemplateOptions(architecture) {
  const existing = new Set(datasetInferenceTemplates(architecture).map((item) => item.dataset_key));
  const options = state.datasets
    .filter((item) => item.key !== "custom" && !existing.has(item.key))
    .map((item) => option(item.key, item.name, false));
  return options.length ? options.join("") : `<option value="">Нет доступных датасетов</option>`;
}

async function renderAutomationPage() {
  const snapshot = await apiJson("/automation");
  const rules = Object.fromEntries(
    (snapshot.rules || []).map((rule) => [automationRuleKey(rule.dataset_key, rule.architecture), rule])
  );
  const headers = [
    "<th>Датасет</th>",
    ...(snapshot.models || []).map((model) => `<th><span class="automation-model-name">${escapeHtml(model.display_name)}</span></th>`),
  ].join("");
  const rows = (snapshot.datasets || []).map((dataset) => `
    <tr>
      <td class="automation-dataset">
        <strong>${escapeHtml(dataset.name)}</strong>
        <small>${formatDate(dataset.updated_at) || "нет даты"} · ${escapeHtml(shortVersion(dataset.version))}</small>
      </td>
      ${(snapshot.models || []).map((model) => {
        const rule = rules[automationRuleKey(dataset.key, model.architecture)] || {};
        return `
          <td
            class="automation-cell"
            data-dataset-key="${escapeAttr(dataset.key)}"
            data-architecture="${escapeAttr(model.architecture)}"
          >
            <button
              class="automation-pill training ${rule.training_enabled ? "active" : ""}"
              data-kind="training"
              data-enabled="${rule.training_enabled ? "true" : "false"}"
              type="button"
            >обучение</button>
            <button
              class="automation-pill pseudo ${rule.pseudo_markup_enabled ? "active" : ""}"
              data-kind="pseudo"
              data-enabled="${rule.pseudo_markup_enabled ? "true" : "false"}"
              type="button"
            >разметка</button>
            <div class="automation-status">
              ${automationStatus("обучение", rule.training_status)}
              ${automationStatus("разметка", rule.pseudo_markup_status)}
            </div>
          </td>
        `;
      }).join("")}
    </tr>
  `).join("");
  renderShell(`
    <section class="hero compact-hero">
      <h1>Автоматизация</h1>
      <p>Автозапуск обучения и псевдоразметки для актуальных версий датасетов MLMarkup</p>
    </section>
    <section class="switch-row">
      <button
        class="automation-toggle-pill ${snapshot.enabled ? "active" : ""}"
        id="automation-toggle"
        type="button"
        data-enabled="${snapshot.enabled ? "true" : "false"}"
      >Автоматизация<br>включена</button>
      <button class="secondary" id="refresh-automation" type="button">Обновить</button>
    </section>
    <section class="panel">
      <div class="automation-table-wrap">
        <table class="automation-table">
          <thead><tr>${headers}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `);
  document.getElementById("automation-toggle").addEventListener("click", async (event) => {
    const enabled = event.currentTarget.dataset.enabled === "true";
    if (!enabled) {
      await apiJson("/automation/enabled", {
        method: "PUT",
        body: { enabled: true },
      });
      renderAutomationPage();
      return;
    }
    showAutomationDisableModal();
  });
  document.getElementById("refresh-automation").addEventListener("click", renderAutomationPage);
  for (const button of document.querySelectorAll(".automation-pill")) {
    button.addEventListener("click", async (event) => {
      const cell = event.currentTarget.closest(".automation-cell");
      const trainingButton = cell.querySelector("[data-kind='training']");
      const pseudoButton = cell.querySelector("[data-kind='pseudo']");
      const kind = event.currentTarget.dataset.kind;
      const trainingEnabled = kind === "training"
        ? event.currentTarget.dataset.enabled !== "true"
        : trainingButton.dataset.enabled === "true";
      const pseudoEnabled = kind === "pseudo"
        ? event.currentTarget.dataset.enabled !== "true"
        : pseudoButton.dataset.enabled === "true";
      await apiJson("/automation/rules", {
        method: "PUT",
        body: {
          dataset_key: cell.dataset.datasetKey,
          architecture: cell.dataset.architecture,
          training_enabled: trainingEnabled,
          pseudo_markup_enabled: pseudoEnabled,
        },
      });
      renderAutomationPage();
    });
  }
}

async function renderQueuePage() {
  const snapshot = await apiJson("/queues");
  const jobs = snapshot.jobs || mergedQueueJobs(snapshot);
  renderShell(`
    <section class="hero compact-hero">
      <h1>В процессе</h1>
      <p>Единая очередь обучения и инференса</p>
    </section>
    <section class="switch-row">
      <label class="switch">
        <input id="training-toggle" type="checkbox" ${snapshot.training_enabled ? "checked" : ""}>
        обучение ${snapshot.training_enabled ? "включено" : "выключено"}
      </label>
      <label class="switch">
        <input id="inference-toggle" type="checkbox" ${snapshot.inference_enabled ? "checked" : ""}>
        инференс ${snapshot.inference_enabled ? "включен" : "выключен"}
      </label>
      <button class="secondary" id="refresh-queues" type="button">Обновить</button>
    </section>
    <section class="panel">
      <h2>Очередь заданий</h2>
      ${renderJobQueue(jobs)}
    </section>
  `);
  document.getElementById("training-toggle").addEventListener("change", (event) => updateQueueEnabled("training", event.currentTarget.checked));
  document.getElementById("inference-toggle").addEventListener("change", (event) => updateQueueEnabled("inference", event.currentTarget.checked));
  document.getElementById("refresh-queues").addEventListener("click", renderQueuePage);
  bindQueueActions();
  scheduleProgressRefresh(renderQueuePage, hasRunningJobs(jobs));
}

function mergedQueueJobs(snapshot) {
  return [...(snapshot.inference_jobs || []), ...(snapshot.training_jobs || [])].sort((left, right) => {
    if (left.status !== right.status) return left.status === "running" ? -1 : 1;
    return queuePriority(right) - queuePriority(left)
      || (left.queue_position || 0) - (right.queue_position || 0)
      || String(left.created_at || "").localeCompare(String(right.created_at || ""));
  });
}

function queuePriority(job) {
  if (job.type === "inference" && job.source === "manual") return 40;
  if (job.type === "training" && job.source === "manual") return 30;
  if (job.type === "inference" && job.source === "automation") return 20;
  if (job.type === "training" && job.source === "automation") return 10;
  return 0;
}

function hasRunningJobs(jobs) {
  return (jobs || []).some((job) => job.status === "running");
}

function hasRunningClassResults(payload) {
  return (payload.results || []).some((result) => {
    if (result.status === "running") return true;
    return (result.pseudo_markup_results || []).some((item) => item.status === "running");
  });
}

function renderJobQueue(jobs) {
  return renderTable(
    ["№", "Датасет", "Модель", "Действие", "Создано", "Начало", "Управление"],
    jobs.map((job, index) => `
      <tr class="job-row ${job.type === "inference" ? "job-row-inference" : "job-row-training"}" data-job="${job.id}">
        <td>${index + 1}</td>
        <td>${queueDatasetCell(job)}</td>
        <td>${queueModelCell(job)}</td>
        <td>
          <span class="result-status-badges">
            ${jobTypeBadge(job.type)}
            ${sourceBadge(job.source)}
            ${statusBadge(job.status, job.type, job.progress)}
          </span>
        </td>
        <td>${formatDateTime(job.created_at)}</td>
        <td>${formatDateTime(job.started_at)}</td>
        <td>${queueButtons(job)}</td>
      </tr>
    `)
  );
}

function queueDatasetCell(job) {
  if (job.type === "inference") {
    const inferenceDataset = job.inference_dataset_name || job.dataset_name || "";
    const trainingDataset = job.training_dataset_name || "";
    return `
      <strong>${escapeHtml(inferenceDataset)}</strong>
      ${trainingDataset ? `<small>по обучению: ${escapeHtml(trainingDataset)}</small>` : ""}
    `;
  }
  return `<strong>${escapeHtml(job.dataset_name)}</strong>`;
}

function queueModelCell(job) {
  return `
    <strong>${escapeHtml(job.model_name)}</strong>
    ${job.type === "training" && job.tile_size ? `<small>тайл ${escapeHtml(job.tile_size)}</small>` : ""}
  `;
}

function jobTypeBadge(type) {
  if (type === "inference") return `<span class="badge inference">инференс</span>`;
  return `<span class="badge training">обучение</span>`;
}

function queueButtons(job) {
  const actions = job.actions || [];
  if (!actions.length) {
    return job.source === "automation" ? `<span class="badge neutral">авто</span>` : "";
  }
  return `
    <div class="inline-row">
      ${actions.includes("move_up") ? `<button class="icon-button queue-action" data-action="up" data-job="${job.id}" title="Повысить">↑</button>` : ""}
      ${actions.includes("move_down") ? `<button class="icon-button queue-action" data-action="down" data-job="${job.id}" title="Понизить">↓</button>` : ""}
      ${actions.includes("delete") ? `<button class="icon-button queue-action danger" data-action="delete" data-job="${job.id}" title="Удалить">×</button>` : ""}
      ${job.source === "automation" ? `<span class="badge neutral">авто</span>` : ""}
    </div>
  `;
}

function bindQueueActions() {
  for (const row of document.querySelectorAll(".job-row")) {
    row.addEventListener("click", () => navigate(`#/jobs/${row.dataset.job}`));
  }
  for (const button of document.querySelectorAll(".queue-action")) {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const id = button.dataset.job;
      if (button.dataset.action === "up") await apiJson(`/jobs/${id}/move-up`, { method: "POST" });
      if (button.dataset.action === "down") await apiJson(`/jobs/${id}/move-down`, { method: "POST" });
      if (button.dataset.action === "delete") await apiJson(`/jobs/${id}`, { method: "DELETE" });
      renderQueuePage();
    });
  }
}

async function updateQueueEnabled(queue, enabled) {
  await apiJson(`/queues/${queue}/enabled`, { method: "PUT", body: { enabled } });
  renderQueuePage();
}

async function renderJobPage(jobId) {
  const job = await apiJson(`/jobs/${encodeURIComponent(jobId)}`);
  const configView = jobConfigView(job);
  renderShell(`
    <section class="toolbar">
      <button class="secondary" type="button" id="job-back">← Назад</button>
      <span class="badge neutral">${escapeHtml(job.status)}</span>
    </section>
    <section class="panel">
      <div class="panel-header">
        <h2>Параметры запуска</h2>
        <span class="badge neutral">${escapeHtml(job.model_name)}</span>
      </div>
      <div class="form-grid">
        <label>Датасет<input readonly value="${escapeAttr(job.dataset_name)}"></label>
        <label>MLflow experiment<input readonly value="${escapeAttr(job.mlflow_experiment_name || "")}"></label>
        <label>MLflow run<input readonly value="${escapeAttr(job.mlflow_run_name || "")}"></label>
      </div>
    </section>
    <section class="panel">
      <h2>${escapeHtml(configView.title)}</h2>
      <div class="form-grid">${renderConfigFields(configView.schema, configView.values, true)}</div>
    </section>
  `);
  document.getElementById("job-back").addEventListener("click", () => navigate("#/queue"));
}

function jobConfigView(job) {
  if (job.type === "inference") {
    const template = inferenceTemplateById(job.config?.inference_template_id)
      || inferenceTemplateFor(job.architecture, job.dataset_key);
    return {
      title: "Параметры инференса",
      schema: template.config_schema,
      values: job.config?.inference_template_config || {},
    };
  }
  const template = templateFor(job.architecture, job.dataset_key);
  return {
    title: "Параметры обучения",
    schema: template.config_schema,
    values: job.config || {},
  };
}

async function renderResultsPage() {
  const changes = await apiJson("/results/changes");
  renderShell(`
    <section class="hero compact-hero">
      <h1>Результаты</h1>
      <p>Классы и варианты датасетов из MLMarkup</p>
    </section>
    <section class="grid">
      ${state.classes.map(renderResultClassCard).join("")}
    </section>
    <section class="panel">
      <h2>Последние изменения</h2>
      ${renderResultChangesTable(changes.changes || [], { activeClick: "class" })}
    </section>
  `);
  bindResultChangeRows({ activeClick: "class" });
}

function renderResultClassCard(item) {
  const variants = item.variants && item.variants.length ? item.variants : [item];
  return `
    <article class="card result-card">
      <div>
        <h2>${escapeHtml(item.name)}</h2>
        <p>Варианты датасета</p>
      </div>
      <div class="variant-list">
        ${variants.map((variant) => `
          <a class="variant-link" href="#/results/${encodeURIComponent(variant.key)}">
            <span>${escapeHtml(variant.variant_name || variant.name)}</span>
            <small>${formatDate(variant.updated_at) || "нет данных"}</small>
          </a>
        `).join("")}
      </div>
    </article>
  `;
}

function renderResultChangesTable(changes, options = {}) {
  if (!changes.length) {
    return `<div class="info-box">Изменений пока нет.</div>`;
  }
  const activeClick = options.activeClick || "class";
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>время</th>
            <th>сеть</th>
            <th>датасет</th>
            <th>что сделано</th>
            <th>MLflow</th>
          </tr>
        </thead>
        <tbody>
          ${changes.map((item) => `
            <tr
              class="result-change-row ${item.item_type === "job" ? "active-job-change-row" : ""}"
              data-class-key="${escapeAttr(item.class_key)}"
              data-job-id="${escapeAttr(item.job_id || "")}"
              data-active-click="${escapeAttr(activeClick)}"
            >
              <td>${formatDateTime(item.changed_at)}</td>
              <td>${escapeHtml(item.model_name)}</td>
              <td>${escapeHtml(item.dataset_name)}</td>
              <td>
                <span class="result-status-badges">
                  ${item.type ? jobTypeBadge(item.type) : ""}
                  ${sourceBadge(item.source)}
                  ${item.item_type === "job"
                    ? statusBadge(item.status, item.type, item.progress)
                    : `<span class="badge ok">${escapeHtml(item.action)}</span>`}
                </span>
              </td>
              <td>${item.mlflow_run_url ? `<a href="${escapeAttr(item.mlflow_run_url)}" target="_blank" rel="noreferrer">MLflow</a>` : ""}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function bindResultChangeRows(options = {}) {
  const activeClick = options.activeClick || "class";
  for (const row of document.querySelectorAll(".result-change-row")) {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, button")) return;
      if (row.classList.contains("active-job-change-row") && activeClick === "job" && row.dataset.jobId) {
        navigate(`#/jobs/${encodeURIComponent(row.dataset.jobId)}`);
        return;
      }
      navigate(`#/results/${encodeURIComponent(row.dataset.classKey)}`);
    });
  }
}

async function renderClassResultPage(classKey) {
  const [payload, changes] = await Promise.all([
    apiJson(`/results/classes/${encodeURIComponent(classKey)}`),
    apiJson("/results/changes"),
  ]);
  const activeJobs = (changes.changes || []).filter((item) => (
    item.item_type === "job" && item.class_key === classKey
  ));
  renderShell(`
    <section class="toolbar">
      <button class="secondary" type="button" id="results-back">← Назад</button>
      <span class="badge neutral">обновлено ${formatDate(payload.dataset_updated_at) || "нет данных"}</span>
    </section>
    <section class="hero compact-hero">
      <h1>${escapeHtml(payload.class_name)}</h1>
    </section>
    ${activeJobs.length ? `
      <section class="panel">
        <h2>Запланировано и в процессе</h2>
        ${renderResultChangesTable(activeJobs, { activeClick: "job" })}
      </section>
    ` : ""}
    <section class="panel">
      ${renderResultsTable(payload)}
    </section>
  `);
  document.getElementById("results-back").addEventListener("click", () => navigate("#/results"));
  bindResultChangeRows({ activeClick: "job" });
  for (const button of document.querySelectorAll(".pseudo-button")) {
    button.addEventListener("click", () => showPseudoModal(payload.class_key, button.dataset.result || ""));
  }
  for (const button of document.querySelectorAll(".result-zip-button")) {
    button.addEventListener("click", () => {
      showTrainingResultZipModal(button.dataset.result || "", button.dataset.defaultName || "model_kanopus");
    });
  }
  for (const button of document.querySelectorAll(".pseudo-delete-button")) {
    button.addEventListener("click", () => {
      showPseudoDeleteModal(payload.class_key, button.dataset.pseudo || "", button.dataset.label || "");
    });
  }
  scheduleProgressRefresh(
    () => renderClassResultPage(classKey),
    hasRunningClassResults(payload) || activeJobs.some((item) => ["queued", "running"].includes(item.status))
  );
}

function renderResultsTable(payload) {
  if (!payload.results.length) {
    return `<div class="info-box">Результатов пока нет.</div>`;
  }
  const groups = payload.results.map((item) => {
    const pseudoAction = item.status === "ok"
      ? `<button class="secondary pseudo-button compact-action" type="button" data-result="${item.id}">Сделать псевдоразметку</button>`
      : "";
    const zipAction = item.status === "ok"
      ? `<button class="secondary result-zip-button compact-action" type="button" data-result="${item.id}" data-default-name="${escapeAttr(defaultTrainingZipModelName(item))}">zip</button>`
      : "";
    const pseudoTable = item.pseudo_markup_results.length ? `
      <tr class="result-pseudo-row">
        <td colspan="7">
          ${renderPseudoTable(item)}
        </td>
      </tr>
    ` : "";
    return `
      <tbody class="result-group">
      <tr class="result-main-row">
        <td class="result-model-cell">${escapeHtml(item.model_name)}</td>
        <td>${formatF1Score(item.f1_score)}</td>
        <td>${item.epoch ?? ""}</td>
        <td>${formatDate(item.trained_at)}</td>
        <td>${item.mlflow_run_url ? `<a href="${escapeAttr(item.mlflow_run_url)}" target="_blank" rel="noreferrer">MLflow</a>` : ""}</td>
        <td>${resultStatusView(item.status, item.source, "training", item.progress)}</td>
        <td>
          <div class="result-actions">
            ${pseudoAction}
            ${zipAction}
          </div>
        </td>
      </tr>
      ${pseudoTable}
      </tbody>
    `;
  }).join("");
  return `
    <div class="plain-table-wrap">
      <table class="plain-table results-table">
        <thead>
          <tr>
            <th>модель</th>
            <th>f1 score</th>
            <th>на эпохе</th>
            <th>дата обучения</th>
            <th>MLflow</th>
            <th>статус</th>
            <th>действия</th>
          </tr>
        </thead>
        ${groups}
      </table>
    </div>
  `;
}

function renderPseudoTable(result) {
  const rows = result.pseudo_markup_results.map((item) => `
    <tr>
      <td>${item.scenes_file ? `<a href="${escapeAttr(item.scenes_file.download_url)}">${escapeHtml(imageSourceLabel(item))}</a>` : escapeHtml(imageSourceLabel(item))}</td>
      <td>${item.geojson_file ? `<a href="${escapeAttr(item.geojson_file.download_url)}">${escapeHtml(geojsonDownloadLabel(item.geojson_file))}</a>` : ""}</td>
      <td>${resultStatusView(item.status, item.source, "inference", item.progress)}</td>
      <td>${formatDateTime(item.created_at)}</td>
      <td>
        <button class="icon-button danger pseudo-delete-button" type="button" data-pseudo="${item.id}" data-label="${escapeAttr(item.source_dataset_name)}" title="Удалить">×</button>
      </td>
    </tr>
  `).join("");
  return `
    <table class="nested-table">
      <thead>
        <tr>
          <th>снимки</th>
          <th>псевдоразметка</th>
          <th>статус</th>
          <th>создано</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

function datasetOptionLabel(item) {
  const count = integerOrNull(item.image_count);
  return count === null ? item.name : `${item.name} (${count} снимков)`;
}

function imageFolderOptionLabel(item) {
  const count = integerOrNull(item.image_count);
  return count === null ? item.name : `${item.name} (${count} снимков)`;
}

function imageSourceLabel(item) {
  const count = imageSourceCount(item);
  return count === null ? item.source_dataset_name : `${item.source_dataset_name} (${count})`;
}

function imageSourceCount(item) {
  const ownCount = integerOrNull(item.image_count);
  if (ownCount !== null) return ownCount;
  if (item.dataset_key && item.dataset_key !== "custom") {
    const dataset = state.datasets.find((candidate) => candidate.key === item.dataset_key);
    const count = integerOrNull(dataset?.image_count);
    if (count !== null) return count;
  }
  const folder = state.imageFolders.find((candidate) => candidate.key === item.source_dataset_name);
  return integerOrNull(folder?.image_count);
}

function geojsonDownloadLabel(file) {
  const size = formatFileSize(file.size_bytes);
  return size ? `скачать geojson (${size})` : "скачать geojson";
}

function defaultTrainingZipModelName(result) {
  const dataset = state.datasets.find((item) => item.key === result.dataset_key);
  const annotationFile = String(dataset?.annotation_file || "");
  const filename = annotationFile.split(/[\\/]/).pop() || "";
  const stem = filename.replace(/\.[^.]*$/, "");
  return `${exportModelNamePart(stem)}_kanopus`;
}

function exportModelNamePart(value) {
  const part = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_-]+/g, "")
    .replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, "");
  return part || "model";
}

function isValidExportModelName(value) {
  return /^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$/.test(String(value || "").trim());
}

function formatFileSize(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${Math.round(bytes)} Б`;
  if (bytes < 1024 * 1024) return `${formatDecimal(bytes / 1024)} КБ`;
  return `${formatDecimal(bytes / (1024 * 1024))} МБ`;
}

function formatDecimal(value) {
  const rounded = Math.round(value * 10) / 10;
  return rounded.toLocaleString("ru-RU", { maximumFractionDigits: 1 });
}

function showPseudoModal(classKey, resultId) {
  const datasetOptions = state.datasets
    .filter((item) => item.key !== "custom")
    .map((item) => `<option value="${escapeAttr(item.key)}">${escapeHtml(datasetOptionLabel(item))}</option>`)
    .join("");
  const folderOptions = state.imageFolders
    .map((item) => `<option value="${escapeAttr(item.key)}">${escapeHtml(imageFolderOptionLabel(item))}</option>`)
    .join("");
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Сделать псевдоразметку</h2>
        <form id="pseudo-form" class="form-stack" action="javascript:void(0)" method="post">
          <label>Список снимков из датасета
            <input name="dataset_key" list="pseudo-datasets" autocomplete="off">
            <datalist id="pseudo-datasets">${datasetOptions}</datalist>
          </label>
          <label>Папка со снимками
            <input name="image_folder_key" list="pseudo-image-folders" autocomplete="off">
            <datalist id="pseudo-image-folders">${folderOptions}</datalist>
          </label>
          <label>Или свой TXT
            <input name="scenes_txt" type="file" accept=".txt,text/plain">
          </label>
          <div class="inline-row">
            <button class="primary" type="submit">Запустить</button>
            <button class="secondary" type="button" id="modal-close">Отмена</button>
          </div>
        </form>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("pseudo-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const file = form.get("scenes_txt");
    const datasetKey = String(form.get("dataset_key") || "").trim();
    const imageFolderKey = String(form.get("image_folder_key") || "").trim();
    const hasFile = file instanceof File && Boolean(file.name);
    const sourceCount = [Boolean(datasetKey), Boolean(imageFolderKey), hasFile].filter(Boolean).length;
    if (sourceCount === 0) {
      showModal("Ошибка", "Выберите датасет, папку снимков или загрузите txt.", "Понятно");
      return;
    }
    if (sourceCount > 1) {
      showModal("Ошибка", "Выберите только один источник снимков.", "Понятно");
      return;
    }
    const request = new FormData();
    if (resultId) request.set("training_result_id", resultId);
    if (datasetKey) {
      request.set("dataset_key", datasetKey);
    } else if (imageFolderKey) {
      request.set("image_folder_key", imageFolderKey);
    } else if (hasFile) {
      request.set("scenes_txt", file);
    }
    await apiForm(`/results/classes/${encodeURIComponent(classKey)}/pseudo-markup`, request);
    closeModal();
    await renderClassResultPage(classKey);
  });
}

function showPseudoDeleteModal(classKey, pseudoId, label) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Удалить псевдоразметку?</h2>
        <p>Запись и сохраненный GeoJSON будут удалены. ${label ? `Источник снимков: ${escapeHtml(label)}.` : ""}</p>
        <div class="inline-row">
          <button class="danger" type="button" id="pseudo-delete-confirm">Удалить</button>
          <button class="secondary" type="button" id="modal-close">Отмена</button>
        </div>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("pseudo-delete-confirm").addEventListener("click", async () => {
    await apiJson(`/results/pseudo-markup/${encodeURIComponent(pseudoId)}`, { method: "DELETE" });
    closeModal();
    await renderClassResultPage(classKey);
  });
}

function showAutomationDisableModal() {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Отключить автоматизацию?</h2>
        <p>Все незавершенные автоматические результаты будут потеряны, очередь автоматических заданий будет очищена, а текущий автоматический процесс будет остановлен.</p>
        <div class="inline-row">
          <button class="danger" type="button" id="automation-disable-confirm">Отключить</button>
          <button class="secondary" type="button" id="automation-disable-cancel">Отмена</button>
        </div>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("automation-disable-cancel").addEventListener("click", closeModal);
  document.getElementById("automation-disable-confirm").addEventListener("click", async () => {
    const button = document.getElementById("automation-disable-confirm");
    button.disabled = true;
    try {
      await apiJson("/automation/enabled", {
        method: "PUT",
        body: { enabled: false },
      });
      closeModal();
      renderAutomationPage();
    } catch {
      button.disabled = false;
    }
  });
}

function renderConfigFields(schema, values, readonly = false, options = {}) {
  return (schema.fields || []).map((field) => {
    const value = values[field.key];
    const tooltip = configFieldTooltip(field);
    const applyAll = options.applyAll && !readonly
      ? `<button class="secondary field-apply-all" type="button" data-apply-key="${escapeAttr(field.key)}">установить для всех</button>`
      : "";
    const label = `
      <span class="field-help">
        <span class="field-title">${escapeHtml(field.label)}</span>
        <span class="help-icon" tabindex="0" aria-label="${escapeAttr(tooltip)}" data-tooltip="${escapeAttr(tooltip)}">i</span>
      </span>
      <span class="field-key">${escapeHtml(field.key)}</span>
    `;
    if (field.value_type === "boolean") {
      return `
        <label class="config-field-label checkbox-row">
          <input data-config-key="${escapeAttr(field.key)}" data-value-type="${field.value_type}" type="checkbox" ${value ? "checked" : ""} ${readonly ? "disabled" : ""}>
          ${label}
          ${applyAll}
        </label>
      `;
    }
    if (field.value_type === "select") {
      return `
        <label class="config-field-label">${label}
          <select data-config-key="${escapeAttr(field.key)}" data-value-type="${field.value_type}" ${readonly ? "disabled" : ""}>
            ${(field.options || []).map((item) => option(item, item, item === value)).join("")}
          </select>
          ${applyAll}
        </label>
      `;
    }
    const inputType = field.value_type.startsWith("integer") || field.value_type.startsWith("number") ? "number" : "text";
    const step = field.value_type.startsWith("number") ? "any" : "1";
    return `
      <label class="config-field-label">${label}
        <input
          data-config-key="${escapeAttr(field.key)}"
          data-value-type="${field.value_type}"
          type="${inputType}"
          step="${step}"
          value="${value === null || value === undefined ? "" : escapeAttr(String(value))}"
          ${readonly ? "readonly" : ""}
        >
        ${applyAll}
      </label>
    `;
  }).join("");
}

function configFieldTooltip(field) {
  const parts = [];
  if (field.tooltip) parts.push(field.tooltip);
  const allowed = configAllowedRange(field);
  if (allowed) parts.push(`Допустимо: ${allowed}.`);
  if (field.recommended_range) parts.push(`Рекомендуемые значения: ${field.recommended_range}`);
  if (field.options && field.options.length) parts.push(`Варианты: ${field.options.join(", ")}.`);
  return parts.join("\n\n");
}

function configAllowedRange(field) {
  const min = field.min_value;
  const max = field.max_value;
  if (min !== null && min !== undefined && max !== null && max !== undefined) {
    return `${min}..${max}`;
  }
  if (min !== null && min !== undefined) return `не меньше ${min}`;
  if (max !== null && max !== undefined) return `не больше ${max}`;
  if (String(field.value_type || "").includes("null")) return "пусто или числовое значение";
  return "";
}

function collectConfig(form) {
  const config = {};
  for (const element of form.querySelectorAll("[data-config-key]")) {
    const key = element.dataset.configKey;
    const valueType = element.dataset.valueType;
    if (valueType === "boolean") {
      config[key] = element.checked;
      continue;
    }
    const raw = element.value;
    if (raw === "" && valueType.endsWith("-null")) {
      config[key] = null;
      continue;
    }
    if (valueType.startsWith("integer")) {
      config[key] = Number.parseInt(raw, 10);
      continue;
    }
    if (valueType.startsWith("number")) {
      config[key] = Number.parseFloat(raw);
      continue;
    }
    config[key] = raw;
  }
  return config;
}

function selectedTemplate() {
  const byId = state.templates.find((item) => item.id === state.selectedTemplateId);
  if (byId) return byId;
  return baseTemplates().find((item) => item.architecture === state.selectedArchitecture) || state.templates[0];
}

function selectedInferenceTemplate() {
  const byId = state.inferenceTemplates.find((item) => item.id === state.selectedInferenceTemplateId);
  if (byId) return byId;
  return baseInferenceTemplates().find((item) => item.architecture === state.selectedArchitecture)
    || state.inferenceTemplates[0];
}

function inferenceTemplateById(templateId) {
  if (!templateId) return null;
  return state.inferenceTemplates.find((item) => item.id === templateId) || null;
}

function templateFor(architecture, datasetKey = null) {
  const datasetTemplate = datasetKey && datasetKey !== "custom"
    ? state.templates.find((item) => item.architecture === architecture && item.dataset_key === datasetKey && item.is_active)
    : null;
  return datasetTemplate || baseTemplates().find((item) => item.architecture === architecture) || state.templates[0];
}

function inferenceTemplateFor(architecture, datasetKey = null) {
  const datasetTemplate = datasetKey && datasetKey !== "custom"
    ? state.inferenceTemplates.find((item) => item.architecture === architecture && item.dataset_key === datasetKey && item.is_active)
    : null;
  return datasetTemplate
    || baseInferenceTemplates().find((item) => item.architecture === architecture)
    || state.inferenceTemplates[0];
}

function baseTemplates() {
  return state.templates
    .filter((item) => !item.dataset_key)
    .sort((left, right) => left.display_name.localeCompare(right.display_name, "ru"));
}

function baseInferenceTemplates() {
  return state.inferenceTemplates
    .filter((item) => !item.dataset_key)
    .sort((left, right) => left.display_name.localeCompare(right.display_name, "ru"));
}

function datasetTemplates(architecture) {
  return state.templates
    .filter((item) => item.dataset_key && item.architecture === architecture)
    .sort((left, right) => templateTitle(left).localeCompare(templateTitle(right), "ru"));
}

function datasetInferenceTemplates(architecture) {
  return state.inferenceTemplates
    .filter((item) => item.dataset_key && item.architecture === architecture)
    .sort((left, right) => templateTitle(left).localeCompare(templateTitle(right), "ru"));
}

function templateTitle(template) {
  return template.dataset_name || template.display_name;
}

function latestExperimentId() {
  if (!state.experiments.length) return "";
  return state.experiments.reduce((latest, item) => {
    const latestNumber = Number(latest.experiment_id);
    const itemNumber = Number(item.experiment_id);
    if (!Number.isFinite(itemNumber)) return latest;
    if (!Number.isFinite(latestNumber) || itemNumber > latestNumber) return item;
    return latest;
  }, state.experiments[0]).experiment_id;
}

function renderTable(headers, rows) {
  if (!rows.length) return `<div class="info-box">Очередь пуста.</div>`;
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${headers.map((item) => `<th>${escapeHtml(item)}</th>`).join("")}</tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
  `;
}

function statusView(status) {
  if (status === "running") return `<span class="status-icon running"><span class="spinner">↻</span>в процессе</span>`;
  if (status === "ok") return `<span class="status-icon ok">✓ ОК</span>`;
  if (status === "error") return `<span class="status-icon error">× ошибка</span>`;
  if (status === "cancelled") return `<span class="badge neutral">отменено</span>`;
  return `<span class="badge neutral">${escapeHtml(status || "")}</span>`;
}

function resultStatusView(status, source, type, progress) {
  return `
    <span class="result-status-badges">
      ${sourceBadge(source)}
      ${statusBadge(status, type, progress)}
    </span>
  `;
}

function sourceBadge(source) {
  if (source === "automation") return `<span class="badge automation">auto</span>`;
  return `<span class="badge manual">manual</span>`;
}

function statusBadge(status, type, progress) {
  if (status === "running") return `<span class="badge warning"><span class="spinner">↻</span> ${runningProgressLabel(type, progress)}</span>`;
  if (status === "ok") return `<span class="badge ok">✓ ОК</span>`;
  if (status === "error") return `<span class="badge error">× ошибка</span>`;
  if (status === "cancelled") return `<span class="badge neutral">отменено</span>`;
  if (status === "queued") return `<span class="badge neutral">в очереди</span>`;
  return `<span class="badge neutral">${escapeHtml(status || "")}</span>`;
}

function runningProgressLabel(type, progress) {
  if (!progress) return "в процессе";
  const current = integerOrNull(progress.current);
  const total = integerOrNull(progress.total);
  const elapsed = integerOrNull(progress.elapsed_minutes);
  if (type === "inference" && current !== null && total !== null) {
    return `${current}/${total} снимков`;
  }
  if (type === "training" && current !== null) {
    return elapsed !== null ? `${current} (${elapsed}м)` : `${current}`;
  }
  return "в процессе";
}

function integerOrNull(value) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : null;
}

function automationRuleKey(datasetKey, architecture) {
  return `${datasetKey}|||${architecture}`;
}

function automationStatus(label, status) {
  if (!status) return "";
  return `<span class="badge ${statusClass(status)}">${escapeHtml(label)}: ${escapeHtml(statusLabel(status))}</span>`;
}

function statusClass(status) {
  if (status === "ok") return "ok";
  if (status === "running") return "warning";
  if (status === "error") return "error";
  return "neutral";
}

function statusLabel(status) {
  if (status === "running") return "в процессе";
  if (status === "ok") return "ОК";
  if (status === "error") return "ошибка";
  if (status === "cancelled") return "отменено";
  return status || "";
}

function shortVersion(version) {
  if (!version) return "нет версии";
  return version.replace(/^git:/, "").replace(/^fs:/, "").slice(0, 8);
}

function option(value, label, selected) {
  return `<option value="${escapeAttr(value)}" ${selected ? "selected" : ""}>${escapeHtml(label)}</option>`;
}

async function apiJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    method: options.method || "GET",
    credentials: "same-origin",
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (response.status === 401 && !options.authOptional) {
    state.user = null;
    renderLogin();
    throw new Error("auth");
  }
  if (!response.ok) {
    const message = await errorMessage(response);
    showModal("Ошибка", message, "Понятно");
    throw new Error(message);
  }
  return response.json();
}

async function apiForm(path, form) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    credentials: "same-origin",
    body: form,
  });
  if (response.status === 401) {
    state.user = null;
    renderLogin();
    throw new Error("auth");
  }
  if (!response.ok) {
    const message = await errorMessage(response);
    showModal("Ошибка", message, "Понятно");
    throw new Error(message);
  }
  return response.json();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadFilename(response) {
  const header = response.headers.get("content-disposition") || "";
  const utfMatch = header.match(/filename\*=UTF-8''([^;]+)/i);
  if (utfMatch) {
    try {
      return decodeURIComponent(utfMatch[1]);
    } catch {
      return utfMatch[1];
    }
  }
  const match = header.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : "";
}

async function errorMessage(response) {
  try {
    const payload = await response.json();
    return payload.detail || `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

function showModal(title, text, buttonText, onClose = null) {
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(text)}</p>
        <button class="primary" type="button" id="modal-ok">${escapeHtml(buttonText)}</button>
      </section>
    </div>
  `;
  paintModal();
  document.getElementById("modal-ok").addEventListener("click", async () => {
    state.modal = null;
    paintModal();
    if (onClose) {
      await onClose();
      return;
    }
  });
}

function closeModal() {
  state.modal = null;
  paintModal();
}

function paintModal() {
  const modalRoot = document.getElementById("modal-root");
  if (modalRoot) {
    modalRoot.innerHTML = state.modal || "";
  }
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return `${pad(date.getHours())}:${pad(date.getMinutes())} ${pad(date.getDate())}.${pad(date.getMonth() + 1)}`;
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return `${pad(date.getDate())}.${pad(date.getMonth() + 1)}`;
}

function formatF1Score(value) {
  if (value === null || value === undefined || value === "") return "";
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(2) : escapeHtml(String(value));
}

function pad(value) {
  return String(value).padStart(2, "0");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}
