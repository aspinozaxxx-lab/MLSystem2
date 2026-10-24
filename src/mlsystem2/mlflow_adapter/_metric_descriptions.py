"""Русский справочник существующих метрик; численные расчёты здесь не выполняются."""

from __future__ import annotations

from html import escape
from typing import Iterable
from urllib.parse import quote


VERSION = "1"
HTML_PATH = "reports/metric_descriptions.html"
JSON_PATH = "reports/metric_descriptions.json"
NOTE_START = "<!-- grovika-metric-descriptions:start -->"
NOTE_END = "<!-- grovika-metric-descriptions:end -->"


def _entry(meaning: str, purpose: str, unit: str = "доля 0–1", direction: str = "больше — лучше") -> dict[str, str]:
    return {"meaning": meaning, "purpose": purpose, "unit": unit, "direction": direction}


SCORES = {
    "f1": ("F1 = 2×TP/(2×TP+FP+FN)", "Совместная оценка точности и полноты; выявляет пропуски и лишние выделения."),
    "precision": ("Точность = TP/(TP+FP)", "Какая доля найденного верна; снижение означает больше ложных срабатываний."),
    "recall": ("Полнота = TP/(TP+FN)", "Какая доля размеченного найдена; снижение означает больше пропусков."),
    "iou": ("IoU = TP/(TP+FP+FN)", "Доля пересечения прогнозируемой и эталонной области в их объединении."),
}

SAMPLING = {
    "draws": ("Всего предъявлений train-тайлов за текущую эпоху, включая повторы.", "Оценка фактической длины эпохи; это не число уникальных тайлов."),
    "foreground_draws": ("Предъявления информативных тайлов: с объектами или частичной отрицательной разметкой.", "Проверка бюджета обучения на объектах с учётом повторов."),
    "background_draws": ("Предъявления полностью фоновых train-тайлов в текущей эпохе.", "Контроль обучения на пустых участках и подавления мусора."),
    "background_fraction": ("background_draws / draws текущей эпохи.", "Фактическая доля фона; не вес тайла и не доля фоновых пикселей."),
    "unique_foreground": ("Различные информативные train-тайлы текущей эпохи.", "Контроль охвата положительных и частично отрицательных примеров без повторов."),
    "unique_background": ("Различные полностью фоновые train-тайлы текущей эпохи.", "Контроль разнообразия фона; внутри эпохи фон не повторяется."),
    "positive_repeats_max_observed": ("Максимальное число предъявлений одного информативного тайла в эпохе.", "Проверка ограничения повторов; много повторов повышает риск переобучения."),
    "hard_background_draws": ("Часть фоновой квоты, выбранная как сложный фон: явные отрицательные примеры и ошибки train.", "Контроль усиленного обучения против ложных срабатываний; входит в background_draws."),
    "background_groups": ("Число пространственных групп фона, представленных в текущей эпохе.", "Контроль географического разнообразия; группа — снимок и зона размером четыре тайла."),
    "next_background_draws": ("Запланированная квота фоновых тайлов следующей эпохи после обратной связи validation.", "Показывает решение балансировщика, а не уже обработанные тайлы."),
    "next_background_fraction": ("Запланированная доля фона следующей эпохи после обратной связи validation.", "Наблюдение за автоматической корректировкой; сравнивать с текущей background_fraction."),
}

STATIC = {
    "train/epochs_total": _entry("Число завершённых эпох обучения.", "Проверка фактической длительности и условий остановки.", "эпохи", "справочное значение"),
    "train/training_time_sec": _entry("Время внутри train: обучение, проверки и сохранение весов, включая итоговые диагностики, если выполнялись.", "Сравнение затрат; не включает предварительную подготовку датасета и весь конвейер.", "секунды", "меньше — быстрее при сопоставимом качестве"),
    "train/epoch_time_sec": _entry("Время train и выполненной validation текущей эпохи до записи её результата.", "Поиск замедления; эпохи с validation и без неё имеют разный состав работ.", "секунды", "меньше — быстрее при одинаковом объёме"),
    "train/learning_rate": _entry("Скорость обновления весов оптимизатором в текущей эпохе, до шага планировщика.", "Проверка работы планировщика и реакции на застой обучения.", "коэффициент", "нет универсального лучшего значения"),
    "train/stopped_early": _entry("1 — обработана команда остановки с сохранением лучших весов; 0 — такой команды не было.", "В текущем коде это признак команды остановки, а не индикатор срабатывания patience. Остановка по patience или времени также может дать 0.", "0 или 1", "индикатор, не качество"),
    "train/peak_vram_bytes": _entry("Пиковая выделенная тензорам память CUDA по счётчику PyTorch; на CPU — 0.", "Оценка потребности обучения в видеопамяти; не вся память процесса из nvidia-smi.", "байты", "меньше — экономнее при сопоставимой задаче"),
    "val/performed": _entry("0 — validation в эпохе пропущена; 1 — выполнена и записан этот индикатор.", "Не принимать отсутствие val-метрик за нулевое качество. Не все старые варианты пишут этот индикатор.", "0 или 1", "индикатор, не качество"),
    "test/checkpoint_epoch": _entry("Эпоха лучших сохранённых весов, на которых выполнена итоговая test-проверка.", "Связывает test-метрики с конкретным checkpoint, а не с последней эпохой.", "номер эпохи", "справочное значение"),
    "val/empty_pixel_fpr": _entry("Ложно выделенные пиксели / допустимые пиксели полностью пустых validation-тайлов при пороге 0,5.", "Главный сигнал мусора на фоне для автоматической доли фона next-gen3/4; не FP/(FP+TP).", direction="меньше — лучше"),
    "val/empty_tile_fpr": _entry("Доля пустых validation-тайлов, где FP ≥ max(4, ceil(N/10000)); N — число допустимых пикселей тайла, порог 0,5.", "Показывает распространённость мусора по территории; единичные ложные пиксели ниже этого допуска тайл ошибочным не делают.", direction="меньше — лучше"),
    "val/foreground_recall": _entry("Пиксели объектов, найденные как любой foreground / все допустимые размеченные пиксели объектов в полной validation, порог 0,5.", "Защита от пропуска объектов при увеличении доли фона в next-gen3/4. Правильность типа multiclass здесь не проверяется."),
    "train/best_macro_pixel_f1": _entry("Максимальная macro pixel F1 среди всех проверенных эпох.", "Лучший достигнутый macro score. Не обязательно относится к best.pt, если checkpoint выбран по loss."),
}


def describe_metric(name: str, context: dict[str, str]) -> dict[str, str]:
    """Расшифровать точное имя; неизвестные метрики нельзя молча выдумывать."""
    variant = context.get("pipeline_variant", "legacy")
    multiclass = context.get("task") == "multiclass"
    if name in STATIC:
        return dict(STATIC[name])
    if name.endswith("_ema") and name.removesuffix("_ema") in STATIC:
        base = STATIC[name.removesuffix("_ema")]
        return {**base, "meaning": base["meaning"] + " Сглаживание: 0,3×текущее + 0,7×предыдущее; первая оценка без сглаживания.",
                "purpose": "Устойчивый к колебаниям сигнал для автоматической корректировки фона."}
    if name.startswith("sampling/") and name.split("/", 1)[1] in SAMPLING:
        field = name.split("/", 1)[1]
        meaning, purpose = SAMPLING[field]
        return _entry(meaning, purpose, "доля 0–1" if "fraction" in field else "количество", "справочное значение, не качество")
    if name in {"train/loss", "val/loss", "test/loss"}:
        split = {"train": "обучающей", "val": "проверочной", "test": "итоговой тестовой"}[name.split("/")[0]]
        extra = " Включает дополнительный KL-штраф согласованности; validation/test его не включают." if name == "train/loss" and variant in {"next_gen4_consistency_v1", "next_gen4_consistency_v2"} else ""
        return _entry(f"Средняя функция потерь на {split} выборке по настроенному loss и весам." + extra,
                      "Контроль сходимости. Сравнивать только при одинаковых loss, весах и выборке; это не процент ошибок и не F1.",
                      "безразмерная величина", "меньше — лучше")
    if name in {"train/region_loss", "val/region_loss", "train/boundary_loss", "val/boundary_loss"}:
        boundary = "boundary" in name
        meaning = ("Диагностический Dice-loss границы внутри foreground." if boundary else "Диагностический Dice-loss общей площади объектов.") if variant == "object_f1_v2" else (
            "Вспомогательный loss границы: среднее BCE и Dice-loss." if boundary else "Loss площади объектов основного выхода; формула задана конфигурацией.")
        return _entry(meaning, "Проверка обучения границ отдельно от площади. В object f1 v2 эти диагностики не являются слагаемыми общего loss.", "безразмерная величина", "меньше — лучше")
    if name in {"val/best_threshold", "val/best_pixel_threshold", "test/threshold", "train/best_confidence_threshold"}:
        scope = "пиксельной метрики" if name == "val/best_pixel_threshold" else "выбранного критерия качества"
        return _entry(f"Порог уверенности для {scope}; у train — порог выбранного checkpoint, у test — применённый при test.",
                      "Позволяет воспроизвести оценку. Слово best в имени не означает поиск: при фиксированном режиме берётся заданный порог. Не обязательно совпадает с порогом рабочего инференса.",
                      "вероятность 0–1", "настройка, не качество")
    if name in {"train/best_quality_f1", "train/best_object_f1", "train/best_threshold_pixel_f1"}:
        object_score = name == "train/best_object_f1"
        kind = "объектовая" if object_score else "пиксельная" if name.endswith("threshold_pixel_f1") else "выбранного критерия качества"
        # Для object f1 pixel_f1 не записан в checkpoint_selection: адаптер сохраняет максимум истории.
        selected = (variant in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2"}
                    and context.get("checkpoint_selection_metric", "val_loss") == "val_loss") or object_score
        return _entry(f"F1 ({kind}) {'выбранного best.pt' if selected else 'с максимальным значением в истории validation'}.",
                      "Итоговая сводка validation, не независимый test. При выборе по loss F1 лучших весов может быть ниже максимума графика; сверять reports/checkpoint_selection.json.")
    parts = name.split("/")
    if len(parts) == 4 and parts[:2] == ["val", "class"] and parts[3] in SCORES:
        base, purpose = SCORES[parts[3]]
        return _entry(f"{base} для пикселей типа «{parts[2]}» в validation, один тип против остальных.", purpose + " Позволяет найти слабые типы, скрытые средним score.")
    if len(parts) == 4 and parts[:2] == ["val", "scene"]:
        field = parts[3].removeprefix("pixel_").removeprefix("fixed_0_5_")
        if field in SCORES:
            base, purpose = SCORES[field]
            threshold = "фиксированном пороге 0,5" if parts[3].startswith("fixed_0_5_") else "пороге оценки этой эпохи"
            return _entry(f"{base} на validation-окнах сцены «{parts[2]}» при {threshold}.", purpose + " Это не качество на всём снимке, если проверяется только часть его окон.")
    if len(parts) == 2 and parts[0] in {"val", "test"}:
        field = parts[1]
        aliases = {"best_threshold_pixel_f1": "pixel_f1", "best_threshold_object_f1": "object_f1",
                   "best_threshold_object_precision": "object_precision", "best_threshold_object_recall": "object_recall",
                   "best_threshold_precision": "quality_precision", "best_threshold_recall": "quality_recall"}
        field = aliases.get(field, field)
        for prefix in ("fixed_0_5_pixel_", "foreground_pixel_", "macro_pixel_", "micro_pixel_", "pixel_", "object_", "quality_"):
            if not field.startswith(prefix) or field[len(prefix):] not in SCORES:
                continue
            score, purpose = SCORES[field[len(prefix):]]
            aggregation = {
                "fixed_0_5_pixel_": "по пикселям при фиксированном пороге 0,5",
                "foreground_pixel_": "по пикселям объединения всех типов объектов без различения типов",
                "macro_pixel_": "с равным весом каждого присутствующего или предсказанного типа, без фонового класса" if multiclass else "с равным весом каждой проверяемой сцены",
                "micro_pixel_": "по суммарным TP/FP/FN всех типов без фонового класса" if multiclass else "по суммарным пиксельным TP/FP/FN всех проверяемых окон",
                "pixel_": "как macro по типам без фонового класса" if multiclass else "по суммарным пиксельным TP/FP/FN проверяемых окон",
                "object_": "по объектам: взаимно-однозначное сопоставление при IoU ≥ 0,5, после разделения объектов выбранного профиля",
                "quality_": "по выбранному критерию: объекты либо пиксели; для next-gen v1 — среднее по сценам, для multiclass — macro по типам",
            }[prefix]
            split = "validation" if parts[0] == "val" else "test лучших сохранённых весов"
            note = " Проверять порог в val/best_pixel_threshold или val/best_threshold; он может быть фиксированным, несмотря на best в имени."
            if parts[0] == "test":
                note = " Порог — test/threshold; test не участвует в выборе весов. Разбиение test внутри датасета само по себе не гарантирует независимость территории."
            if prefix == "macro_pixel_":
                note += " Macro F1 — среднее F1, а не F1 от средних precision и recall; отсутствующие и в разметке, и в прогнозе типы пропускаются."
            return _entry(f"{score} {aggregation}; {split}.", purpose + note)
    raise ValueError(f"Нет описания MLflow-метрики: {name}")


def build_catalog(names: Iterable[str], context: dict[str, str]) -> dict[str, object]:
    """Описать только реально записанные ключи, включая типы и сцены."""
    variant = context.get("pipeline_variant", "legacy")
    selection = context.get("checkpoint_selection_metric")
    if variant in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2"} and selection in {None, "val_loss"}:
        checkpoint_note = "Лучшие веса этого запуска выбираются по минимуму val/loss, не по максимальной F1."
    elif selection == "val/object_f1" or variant in {"object_f1", "object_f1_v2"}:
        checkpoint_note = "Лучшие веса этого запуска выбираются по максимуму объектовой F1."
    else:
        checkpoint_note = "Лучшие веса выбираются по F1 заданного критерия качества; исторический тег quality_f1 сохраняет это правило. Точный выбор — в reports/checkpoint_selection.json, если отчёт доступен."
    return {
        "version": VERSION,
        "language": "ru",
        "context": context,
        "notes": [
            "Значение F1 0,65 означает 65 %. Доли не записываются как проценты.",
            "TP — верные выделения; FP — лишние выделения; FN — пропуски. Для object-метрик считаются объекты, для pixel — пиксели.",
            "Имена и числа метрик не изменены. Алиасы могут показывать один и тот же результат.",
            "train — обучение; val — выбор весов и настройка; test — итоговая внутренняя проверка. Оценка тестером и качество рабочего инференса — отдельные измерения.",
            checkpoint_note,
            "next-gen2/3/4 оценивают при пороге 0,5, а их рабочий Gaussian-инференс по умолчанию использует >0,9. Поэтому F1 обучения и F1 готовых полигонов не равны автоматически.",
            "Сравнивать запуски корректно только при одинаковых разбиении, разметке, пороге и правилах обработки. Отсутствующая метрика не равна нулю.",
            "В legacy nodata входит в фон; новые профили с valid mask исключают невалидные пиксели. next-gen2 может сохранять старую обработку nodata. Точные условия — в конфигурации и split manifest запуска.",
        ],
        "metrics": {name: describe_metric(name, context) for name in sorted(set(names))},
    }


def render_html(catalog: dict[str, object]) -> str:
    context = catalog["context"]
    rows = []
    for name, item in catalog["metrics"].items():
        tooltip = " ".join(item.values())
        rows.append("<tr><th scope=\"row\"><code title=\"" + escape(tooltip, quote=True) + "\">" + escape(name) + "</code></th>"
                    + "".join(f"<td>{escape(item[key])}</td>" for key in ("meaning", "purpose", "unit", "direction")) + "</tr>")
    notes = "".join(f"<li>{escape(note)}</li>" for note in catalog["notes"])
    labels = {"pipeline_variant": "Конвейер", "task": "Задача", "quality_metric": "Критерий качества",
              "checkpoint_selection_metric": "Выбор лучших весов"}
    details = "; ".join(f"{labels.get(key, key)}: {value}" for key, value in context.items())
    return """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Справочник метрик Гровики</title>
<style>body{font:15px/1.5 system-ui,sans-serif;color:#182533;margin:24px;background:#fff}
h1{font-size:24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #cbd5e1;padding:10px;text-align:left;vertical-align:top}
thead th{background:#eaf2fa;position:sticky;top:0}tbody th{max-width:320px}code{overflow-wrap:anywhere;cursor:help}tbody tr:nth-child(even){background:#f6f8fa}
.hint{background:#eaf2fa;padding:12px;border-radius:6px}li{margin:5px 0}</style></head><body>
<h1>Справочник метрик Гровики</h1><p class="hint">Наведите курсор на название метрики для краткой подсказки.
 Для поиска нажмите Ctrl+F. Все описания также приведены в таблице; подключение к внешним сайтам не требуется.</p>
""" + f"<p>{escape(details)}</p><details><summary>Как сравнивать метрики: пороги, выборки и выбор лучших весов</summary><ul>{notes}</ul></details><br>" + """<table><thead><tr><th>Метрика</th><th>Что означает</th>
<th>Зачем нужна и как читать</th><th>Единицы</th><th>Ориентир</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></body></html>"


def update_run_note(note: str, experiment_id: str, run_id: str) -> str:
    """Заменить только наш блок, сохранив пользовательское описание."""
    route = f"#/experiments/{quote(experiment_id, safe='')}/runs/{quote(run_id, safe='')}/artifacts/{HTML_PATH}"
    block = (f"{NOTE_START}\n### Справочник метрик Гровики\n\n"
             f"[Открыть описания всех метрик]({route}) — смысл, назначение, единицы и подсказки при наведении внутри справочника.\n\n"
             f"Также: **Artifacts → {HTML_PATH}**. Машиночитаемая версия: `{JSON_PATH}`. "
             "F1 0,65 = 65 %; validation, внутренний test и тестер — разные оценки.\n"
             f"{NOTE_END}")
    start = note.find(NOTE_START)
    end = note.find(NOTE_END, start) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return note[:start] + block + note[end + len(NOTE_END):]
    return note + ("\n\n" if note else "") + block
