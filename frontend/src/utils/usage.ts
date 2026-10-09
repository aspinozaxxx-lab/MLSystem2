import type { UsageConfig } from "../api/types";

export const usagePages = {
  home: "Главная", start: "Запуск обучения", queue: "Очередь",
  training_templates: "Шаблоны обучения", inference_templates: "Шаблоны инференса",
  automation: "Автоматизация", classes: "Классы", dataset_editor: "Редактор датасета",
  model_export: "Экспорт моделей", scene_export: "Список сцен",
  test_create: "Создание тестовых разметок", test_catalog: "Тестовые разметки",
  test_editor: "Редактор тестовой разметки", results: "Результаты",
  dataset_results: "Результаты датасета", job: "Задание",
  pseudo_view: "Просмотр псевдоразметки", pseudo_compare: "Сравнение псевдоразметок",
  test_f1: "Просмотр тестового F1", news: "Новости", feedback: "Обращения",
} as const;

export type UsagePage = keyof typeof usagePages;

export const usageActions: Record<string, string> = {
  training_start: "Запуск обучения", training_continue: "Продолжение обучения",
  pseudo_create: "Создание псевдоразметки", test_f1_calculate: "Расчёт тестового F1",
  job_cancel: "Отмена задания", job_finish: "Завершение с лучшими весами",
  queue_move: "Изменение порядка очереди", queue_toggle: "Включение или остановка очереди",
  training_template_create: "Создание шаблона обучения", training_template_save: "Сохранение шаблона обучения",
  training_template_delete: "Удаление шаблона обучения", inference_template_create: "Создание шаблона инференса",
  inference_template_save: "Сохранение шаблона инференса", inference_template_delete: "Удаление шаблона инференса",
  template_assign: "Назначение шаблона классу", class_create: "Создание класса", class_save: "Изменение класса",
  dataset_create: "Создание датасета", dataset_copy: "Копирование датасета", dataset_delete: "Удаление датасета",
  dataset_import: "Импорт разметки", dataset_publish: "Публикация разметки", dataset_rebuild: "Пересборка датасета",
  test_create: "Создание тестовой разметки", test_save: "Изменение тестовой разметки",
  test_delete: "Удаление тестовой разметки", test_optimize: "Оптимизация тестовой разметки",
  primary_select: "Выбор основной сети или разметки", pseudo_delete: "Удаление псевдоразметки",
  model_export: "Экспорт модели", markup_export: "Экспорт разметки", scene_export: "Создание списка сцен",
  file_download: "Скачивание файла", automation_save: "Настройка автоматизации", feedback_send: "Отправка обращения",
};

export function usagePageForRoute(route: readonly string[]): UsagePage {
  const [head, second] = route;
  if (head === "templates") return second === "inference" ? "inference_templates" : "training_templates";
  if (head === "pseudo-markup") return second === "compare" ? "pseudo_compare" : "pseudo_view";
  if (head === "results") return second ? "dataset_results" : "results";
  if (head === "test-markups") return second === "create" ? "test_create" : second ? "test_editor" : "test_catalog";
  return ({ start: "start", queue: "queue", automation: "automation", classes: "classes",
    "dataset-editor": "dataset_editor", "model-export": "model_export", "scene-list-export": "scene_export",
    "test-f1": "test_f1", jobs: "job", news: "news", feedback: "feedback" } as Record<string, UsagePage>)[head] ?? "home";
}

export function usagePageViewed(page: UsagePage): void {
  window.dispatchEvent(new CustomEvent("grovika:page", { detail: page }));
}

type MetricaFunction = ((...args: unknown[]) => void) & { a?: unknown[][]; l?: number };
type MetricaWindow = Window & { ym?: MetricaFunction };
export type UsageTracker = {
  page: (page: UsagePage) => void;
  action: (name: string, status: number, userId: string | null) => void;
  stop: () => void;
};

export function startMetrica(config: UsageConfig): UsageTracker | null {
  const id = config.metrica_counter_id;
  if (!id || id <= 0) return null;
  const target = window as MetricaWindow;
  if (!target.ym) {
    const queued: MetricaFunction = (...args) => {
      const pending = queued.a ??= [];
      if (pending.length < 200) pending.push(args);
    };
    queued.l = Date.now();
    target.ym = queued;
  }
  // tag.js заменяет очередь настоящим обработчиком: всегда читаем актуальный ym.
  const ym = (...args: unknown[]) => {
    try { target.ym?.(...args); } catch { /* Ошибка аналитики не меняет работу Гровики. */ }
  };
  if (!document.querySelector("script[data-grovika-metrica]")) {
    const script = document.createElement("script");
    script.async = true;
    script.src = `https://mc.yandex.ru/metrika/tag.js?id=${id}`;
    script.dataset.grovikaMetrica = "true";
    document.head.appendChild(script);
  }
  // Переходы SPA считаем явно; опрос API не создаёт просмотров.
  ym(id, "init", { defer: true, webvisor: true, clickmap: true, trackLinks: false, accurateTrackBounce: true });
  ym(id, "setUserID", config.metrica_user_id);
  ym(id, "userParams", { grovika_user_id: config.metrica_user_id });
  let active = true;
  let previous = window.location.origin;
  return {
    page(page) {
      if (!active) return;
      const url = `${window.location.origin}/#/usage/${page}`;
      ym(id, "hit", url, { title: `Гровика · ${usagePages[page]}`, referer: previous,
        params: { grovika: { pages: { [config.metrica_user_id]: usagePages[page] } } } });
      previous = url;
    },
    action(name, status, userId) {
      if (!active || !Object.hasOwn(usageActions, name) || userId !== config.metrica_user_id) return;
      const params = { grovika: { features: { [config.metrica_user_id]: {
        [usageActions[name]]: status < 400 ? "Успешно" : "Ошибка",
      } } } };
      // Параметры событий видны и без ручного создания цели в счётчике.
      ym(id, "params", params);
      // Цель означает успешный запрос; параметры отправлены ровно один раз.
      if (status < 400) ym(id, "reachGoal", `grovika_${name}`);
    },
    stop() {
      if (!active) return;
      active = false;
      // Не оставляем события старой сессии в очереди при заблокированном скрипте.
      if (target.ym?.a) target.ym.a = target.ym.a.filter((args) => args[0] !== id);
      ym(id, "destruct");
    },
  };
}
