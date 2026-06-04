const API = "/api/v1";
const root = document.getElementById("app");

const state = {
  user: null,
  links: [],
  datasets: [],
  classes: [],
  models: [],
  templates: [],
  experiments: [],
  selectedArchitecture: "",
  modal: null,
};

window.addEventListener("hashchange", () => render());
init();

async function init() {
  const me = await apiJson("/auth/me", { authOptional: true });
  state.user = me && me.authenticated ? me.username : null;
  await render();
}

async function render() {
  if (!state.user) {
    renderLogin();
    return;
  }
  await ensureSharedData();
  const route = currentRoute();
  if (route[0] === "start") return renderStartPage();
  if (route[0] === "queue") return renderQueuePage();
  if (route[0] === "templates") return renderTemplatesPage();
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

async function ensureSharedData() {
  const [links, datasets, classes, models, templates] = await Promise.all([
    apiJson("/app-links"),
    apiJson("/datasets"),
    apiJson("/classes"),
    apiJson("/models"),
    apiJson("/training-templates"),
  ]);
  state.links = links.links || [];
  state.datasets = datasets.datasets || [];
  state.classes = classes.classes || [];
  state.models = models.models || [];
  state.templates = templates.templates || [];
  if (!state.selectedArchitecture && state.models.length) {
    state.selectedArchitecture = state.models[0].architecture;
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
        <a href="#/templates">Шаблоны обучения</a>
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

function renderHomePage() {
  const linkMap = Object.fromEntries(state.links.map((item) => [item.key, item.url]));
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
    </section>

    <h2 class="section-title">Обучение моделей</h2>
    <section class="grid">
      ${homeTrainingCard("Запуск обучения", "Форма создания training job.", "#/start")}
      ${homeTrainingCard("В процессе", "Очереди обучения и инференса.", "#/queue")}
      ${homeTrainingCard("Шаблоны обучения", "Defaults по архитектурам.", "#/templates")}
      ${homeTrainingCard("Результаты", "Классы, метрики и псевдоразметка.", "#/results")}
    </section>
  `);
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
  const template = selectedTemplate();
  const datasets = state.datasets.map((item) => option(item.key, item.name, false)).join("");
  const models = state.models.map((item) => option(item.architecture, item.display_name, item.architecture === state.selectedArchitecture)).join("");
  const experimentsOptions = [
    `<option value="">Новый experiment</option>`,
    ...state.experiments.map((item) => option(item.experiment_id, item.name, false)),
  ].join("");
  const runName = recommendedRunName();
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
          <label>Новое имя experiment
            <input name="experiment_name" value="MLSystem2">
          </label>
          <label>MLflow run name
            <input name="run_name" value="${escapeAttr(runName)}">
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
          <span class="badge neutral">${escapeHtml(template.display_name)}</span>
        </div>
        <div class="form-grid" id="config-fields">
          ${renderConfigFields(template.config_schema, template.default_config)}
        </div>
      </section>
      <button class="primary" type="submit">Запустить обучение</button>
    </form>
  `);

  const datasetSelect = document.getElementById("dataset-select");
  const uploadPanel = document.getElementById("custom-upload");
  const syncUpload = () => uploadPanel.classList.toggle("hidden", datasetSelect.value !== "custom");
  datasetSelect.addEventListener("change", () => {
    syncUpload();
    document.querySelector("input[name='run_name']").value = recommendedRunName(datasetSelect.value);
  });
  syncUpload();

  document.getElementById("architecture-select").addEventListener("change", async (event) => {
    state.selectedArchitecture = event.currentTarget.value;
    await renderStartPage();
  });
  document.getElementById("start-form").addEventListener("submit", submitStartForm);
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
    mlflow_run_name: String(data.get("run_name") || recommendedRunName()).trim(),
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
  const templateOptions = state.templates.map((item) => option(item.architecture, item.display_name, item.architecture === state.selectedArchitecture)).join("");
  renderShell(`
    <section class="hero compact-hero">
      <h1>Шаблоны обучения</h1>
      <p>Defaults хранятся в Postgres и используются на странице запуска</p>
    </section>
    <section class="panel">
      <div class="toolbar">
        <label>Архитектура
          <select id="template-select">${templateOptions}</select>
        </label>
        <div class="inline-row">
          <span class="badge ${template.source === "manual" ? "warning" : "ok"}">source=${template.source}</span>
          <span class="badge neutral">version=${template.version}</span>
        </div>
      </div>
      <form id="template-form" class="form-stack">
        <div class="form-grid">${renderConfigFields(template.config_schema, template.default_config)}</div>
        <div class="inline-row">
          <button class="primary" type="submit">Сохранить</button>
          <button class="secondary" type="button" id="template-reset">Сбросить</button>
        </div>
      </form>
    </section>
  `);
  document.getElementById("template-select").addEventListener("change", async (event) => {
    state.selectedArchitecture = event.currentTarget.value;
    renderTemplatesPage();
  });
  document.getElementById("template-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await apiJson(`/training-templates/${encodeURIComponent(state.selectedArchitecture)}`, {
      method: "PUT",
      body: { default_config: collectConfig(event.currentTarget) },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
  document.getElementById("template-reset").addEventListener("click", async () => {
    await apiJson(`/training-templates/${encodeURIComponent(state.selectedArchitecture)}`, {
      method: "PUT",
      body: { reset_to_baseline: true },
    });
    await ensureSharedData();
    renderTemplatesPage();
  });
}

async function renderQueuePage() {
  const snapshot = await apiJson("/queues");
  renderShell(`
    <section class="hero compact-hero">
      <h1>В процессе</h1>
      <p>Очереди training и inference</p>
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
      <h2>Очередь обучений</h2>
      ${renderTrainingQueue(snapshot.training_jobs)}
    </section>
    <section class="panel">
      <h2>Очередь инференса</h2>
      ${renderInferenceQueue(snapshot.inference_jobs)}
    </section>
  `);
  document.getElementById("training-toggle").addEventListener("change", (event) => updateQueueEnabled("training", event.currentTarget.checked));
  document.getElementById("inference-toggle").addEventListener("change", (event) => updateQueueEnabled("inference", event.currentTarget.checked));
  document.getElementById("refresh-queues").addEventListener("click", renderQueuePage);
  bindQueueActions();
}

function renderTrainingQueue(jobs) {
  return renderTable(
    ["№", "Датасет", "Модель", "Размер тайла", "Создано", "Начало", "Действия"],
    jobs.map((job, index) => `
      <tr class="job-row" data-job="${job.id}">
        <td>${index + 1}</td>
        <td>${escapeHtml(job.dataset_name)}</td>
        <td>${escapeHtml(job.model_name)}</td>
        <td>${job.tile_size || ""}</td>
        <td>${formatDateTime(job.created_at)}</td>
        <td>${formatDateTime(job.started_at)}</td>
        <td>${queueButtons(job)}</td>
      </tr>
    `)
  );
}

function renderInferenceQueue(jobs) {
  return renderTable(
    ["№", "Датасет обучения", "Модель", "Датасет инференса", "Создано", "Начало", "Действия"],
    jobs.map((job, index) => `
      <tr class="job-row" data-job="${job.id}">
        <td>${index + 1}</td>
        <td>${escapeHtml(job.training_dataset_name || "")}</td>
        <td>${escapeHtml(job.model_name)}</td>
        <td>${escapeHtml(job.inference_dataset_name || "")}</td>
        <td>${formatDateTime(job.created_at)}</td>
        <td>${formatDateTime(job.started_at)}</td>
        <td>${queueButtons(job)}</td>
      </tr>
    `)
  );
}

function queueButtons(job) {
  const running = job.status === "running";
  return `
    <div class="inline-row">
      <button class="icon-button queue-action" data-action="up" data-job="${job.id}" title="Повысить" ${running ? "disabled" : ""}>↑</button>
      <button class="icon-button queue-action" data-action="down" data-job="${job.id}" title="Понизить" ${running ? "disabled" : ""}>↓</button>
      <button class="icon-button queue-action danger" data-action="delete" data-job="${job.id}" title="Удалить">×</button>
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
  const template = state.templates.find((item) => item.architecture === job.architecture) || selectedTemplate();
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
      <h2>Config</h2>
      <div class="form-grid">${renderConfigFields(template.config_schema, job.config, true)}</div>
    </section>
  `);
  document.getElementById("job-back").addEventListener("click", () => navigate("#/queue"));
}

function renderResultsPage() {
  renderShell(`
    <section class="hero compact-hero">
      <h1>Результаты</h1>
      <p>Классы из MLMarkup и Custom</p>
    </section>
    <section class="grid">
      ${state.classes.map((item) => `
        <a class="card active" href="#/results/${encodeURIComponent(item.key)}">
          <div>
            <h2>${escapeHtml(item.name)}</h2>
            <p>Последнее обновление: ${formatDate(item.updated_at) || "нет данных"}</p>
          </div>
          <span class="secondary">Открыть</span>
        </a>
      `).join("")}
    </section>
  `);
}

async function renderClassResultPage(classKey) {
  const payload = await apiJson(`/results/classes/${encodeURIComponent(classKey)}`);
  renderShell(`
    <section class="toolbar">
      <button class="secondary" type="button" id="results-back">← Назад</button>
      <span class="badge neutral">обновлено ${formatDate(payload.dataset_updated_at) || "нет данных"}</span>
    </section>
    <section class="hero compact-hero">
      <h1>${escapeHtml(payload.class_name)}</h1>
    </section>
    <section class="panel">
      ${renderResultsTable(payload)}
    </section>
  `);
  document.getElementById("results-back").addEventListener("click", () => navigate("#/results"));
  for (const button of document.querySelectorAll(".pseudo-button")) {
    button.addEventListener("click", () => showPseudoModal(payload.class_key, button.dataset.result || ""));
  }
}

function renderResultsTable(payload) {
  if (!payload.results.length) {
    return `<div class="info-box">Результатов пока нет.</div>`;
  }
  const rows = payload.results.map((item) => `
    <tr>
      <td>${escapeHtml(item.model_name)}</td>
      <td>${formatF1Score(item.f1_score)}</td>
      <td>${item.epoch ?? ""}</td>
      <td>${formatDate(item.trained_at)}</td>
      <td>${item.mlflow_run_url ? `<a href="${escapeAttr(item.mlflow_run_url)}" target="_blank" rel="noreferrer">MLflow</a>` : ""}</td>
      <td>${statusView(item.status)}</td>
    </tr>
    <tr>
      <td colspan="6">
        ${renderPseudoTable(item)}
      </td>
    </tr>
  `).join("");
  return `
    <div class="plain-table-wrap">
      <table class="plain-table">
        <thead>
          <tr>
            <th>модель</th>
            <th>f1 score</th>
            <th>на эпохе</th>
            <th>дата обучения</th>
            <th>MLflow</th>
            <th>статус</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderPseudoTable(result) {
  const rows = result.pseudo_markup_results.map((item) => `
    <tr>
      <td>${item.scenes_file ? `<a href="${escapeAttr(item.scenes_file.download_url)}"> ${escapeHtml(item.source_dataset_name)}</a>` : escapeHtml(item.source_dataset_name)}</td>
      <td>${item.geojson_file ? `<a href="${escapeAttr(item.geojson_file.download_url)}">скачать geojson</a>` : ""}</td>
      <td>${statusView(item.status)}</td>
      <td>${formatDateTime(item.created_at)}</td>
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
        </tr>
      </thead>
      <tbody>
        ${rows}
        <tr>
          <td colspan="4">
            <button class="secondary pseudo-button" type="button" data-result="${result.id}">Сделать псевдоразметку</button>
          </td>
        </tr>
      </tbody>
    </table>
  `;
}

function showPseudoModal(classKey, resultId) {
  const options = state.datasets.filter((item) => item.key !== "custom").map((item) => `<option value="${escapeAttr(item.key)}">${escapeHtml(item.name)}</option>`).join("");
  state.modal = `
    <div class="modal-backdrop">
      <section class="modal-card">
        <h2>Сделать псевдоразметку</h2>
        <form id="pseudo-form" class="form-stack">
          <label>Список снимков из датасета
            <input name="dataset_key" list="pseudo-datasets" autocomplete="off">
            <datalist id="pseudo-datasets">${options}</datalist>
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
    if ((!file || !(file instanceof File) || !file.name) && !datasetKey) {
      showModal("Ошибка", "Выберите датасет или загрузите txt.", "Понятно");
      return;
    }
    const request = new FormData();
    if (resultId) request.set("training_result_id", resultId);
    if (datasetKey) {
      request.set("dataset_key", datasetKey);
    } else if (file instanceof File && file.name) {
      request.set("scenes_txt", file);
    }
    await apiForm(`/results/classes/${encodeURIComponent(classKey)}/pseudo-markup`, request);
    closeModal();
    renderClassResultPage(classKey);
  });
}

function renderConfigFields(schema, values, readonly = false) {
  return (schema.fields || []).map((field) => {
    const value = values[field.key];
    const title = `${field.label} (${field.key})`;
    const label = `
      <span class="field-help">
        ${escapeHtml(title)}
        <span class="help-icon" title="${escapeAttr(field.tooltip)}">i</span>
      </span>
    `;
    if (field.value_type === "boolean") {
      return `
        <label class="checkbox-row">
          <input data-config-key="${escapeAttr(field.key)}" data-value-type="${field.value_type}" type="checkbox" ${value ? "checked" : ""} ${readonly ? "disabled" : ""}>
          ${label}
        </label>
      `;
    }
    if (field.value_type === "select") {
      return `
        <label>${label}
          <select data-config-key="${escapeAttr(field.key)}" data-value-type="${field.value_type}" ${readonly ? "disabled" : ""}>
            ${(field.options || []).map((item) => option(item, item, item === value)).join("")}
          </select>
        </label>
      `;
    }
    const inputType = field.value_type.startsWith("integer") || field.value_type.startsWith("number") ? "number" : "text";
    const step = field.value_type.startsWith("number") ? "any" : "1";
    return `
      <label>${label}
        <input
          data-config-key="${escapeAttr(field.key)}"
          data-value-type="${field.value_type}"
          type="${inputType}"
          step="${step}"
          value="${value === null || value === undefined ? "" : escapeAttr(String(value))}"
          ${readonly ? "readonly" : ""}
        >
      </label>
    `;
  }).join("");
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
  return state.templates.find((item) => item.architecture === state.selectedArchitecture) || state.templates[0];
}

function recommendedRunName(datasetKey = null) {
  const dataset = state.datasets.find((item) => item.key === (datasetKey || document.querySelector("[name='dataset_key']")?.value)) || state.datasets[0];
  const architecture = state.selectedArchitecture || "model";
  const date = new Date();
  const stamp = `${String(date.getDate()).padStart(2, "0")}${String(date.getMonth() + 1).padStart(2, "0")}`;
  return `${slug(dataset?.name || "dataset")}_${slug(architecture)}_${stamp}`;
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
  return `<span class="badge neutral">${escapeHtml(status || "")}</span>`;
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

function slug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\p{L}\p{N}_-]+/gu, "");
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
