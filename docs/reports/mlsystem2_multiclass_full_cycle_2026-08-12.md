# Полный цикл multiclass для комбинированных классов

Дата проверки: 2026-08-12. Итог: техническая приёмка пройдена — обе разметки опубликованы, все семь архитектур выполнили CUDA smoke, целевой SegFormer B2 завершил ровно 10 эпох, checkpoints сохранены, test/pseudolabel/AOI/QGIS/Triton/Geoalert-контуры проверены.

Минимальный F1 и обязательное предсказание каждого типа в каждом отдельном запуске условиями приёмки не задавались. Качество `flooding` у короткого smoke-обучения низкое, но все loss и метрики конечны, а полный обычный inference предсказал оба типа.

## Ревизии и доставка

| Компонент | Ревизия | Доставка |
| --- | --- | --- |
| MLSystem2, итоговая реализация | `5ea42b111cec6e86c6b2a9faff9e78c385967c37` | workflow `31599925878`, success |
| MLMarkup, опубликованные датасеты | `16c7d4ea5369861f2330e944d1ad36a78c62b388` | workflow `31595098782`, success |
| Исходная ревизия MLMarkup в manifests | `02860855b9d777777e6d4c584261c45529b34f01` | зафиксирована как source revision |
| Код сборщика датасетов | `b79b5dcdb985b0c017808c1053503689ac491166` | использован для финальной сборки |

Ключевая цепочка изменений MLSystem2:

- `04f88db` — основной полный цикл `per_image_multiclass`;
- `e3a7a11` и `b79b5dc` — устранение коллизий, нахлёстов и корректное вычитание геометрий после репроекции;
- `d3c95fa` — manifest включён в артефакты MLflow;
- `a1a31d5` — безопасное сохранение отсутствующих object metrics;
- `da1219f` — пространственный индекс в разрешении конфликтов псевдоразметки;
- `5ea42b1` — родительский `class_id` в канонической multiclass-псевдоразметке.

На сервере `DEPLOYED_COMMIT` совпал с `5ea42b111cec6e86c6b2a9faff9e78c385967c37`. Сначала был развёрнут новый runtime MLSystem2, затем workflow MLMarkup атомарно опубликовал новый формат. Версия БД обновлена до Alembic head `20260812_0016`.

## Датасеты

Оба `main` опубликованы без TXT как `per_image_multiclass`. Manifest `.mlsystem2-dataset.json` содержит версию схемы, классы, source-папки, ревизии и baseline-хеши; schema также повторяется в каждом GeoJSON. Геометрии переведены в CRS TIFF, обрезаны по реальному valid-data footprint и скопированы во все пересекающиеся сцены. Стабильные feature ID/provenance сохранены.

| Датасет | Build ID | Сцен | Per-image features | Состав |
| --- | --- | ---: | ---: | --- |
| Опустынивание и ветровая эрозия | `0febcfce-1f76-487e-afb4-c5bc506137d9` | 75 | 3378 | 1765 `desertification`, 285 `wind_erosion`, 1328 hard negative |
| Переувлажнения и заболачивания | `38a087fb-f0d1-44ce-a6b3-24af0d4bb631` | 16 | 293 | 170 `flooding`, 123 `waterlogging`, 0 hard negative |

Схемы классов:

| Комбинированный класс | Тип | ID | Цвет | Приоритет |
| --- | --- | ---: | --- | ---: |
| Опустынивание и ветровая эрозия | `desertification` | 1 | `#F59E0B` | 100 |
|  | `wind_erosion` | 2 | `#8B5CF6` | 0 |
| Переувлажнения и заболачивания | `flooding` | 1 | `#3B82F6` | 100 |
|  | `waterlogging` | 2 | `#22C55E` | 0 |

Общий hard negative остаётся без class slug и отображается цветом `#EF4444`. Positive вычитается из hard negative. При пересечении положительных типов выигрывает класс с большим приоритетом, а пересечение вычитается из проигравшей геометрии.

Исходные данные первого набора содержали 500 positive и 14 hard-negative объектов опустынивания, 195 positive и 626 валидных hard-negative объектов ветровой эрозии. Ещё 234 объекта ветровой эрозии имели пустую геометрию: все пропущены и перечислены в build/preview warnings. Увеличение количества объектов в per-image результате объясняется пересечением одного source-объекта с несколькими сценами.

Строгая проверка обоих наборов завершилась с `status=ok`: нет повторяющихся feature ID, несовпадающих схем или запрещённых positive/positive и positive/hard-negative пересечений.

Финальный read-only rebuild preview по живым source-папкам подтвердил:

| Dataset key | Source status | Локальные изменения | Конфликты | Replacement |
| --- | --- | ---: | ---: | --- |
| `51be58fb-fbf0-41b1-b06e-f6a66e03a59b` | `current` | 0 | 0 | 75 сцен; 1765/285 positive; 1328 hard negative |
| `dcb092ff-99d0-464d-ab3b-29082385a1c7` | `current` | 0 | 0 | 16 сцен; 170/123 positive; 0 hard negative |

Preview первого набора снова вернул ровно 234 предупреждения о пустых геометриях. Фактический rebuild не запускался, поскольку target уже current и лишний Git-коммит не требовался.

## Реализованный контракт

Новый формат проходит через settings, подготовку сцен и тайлов, обучение, БД, тестовые выборки, метрики, обычную и AOI-псевдоразметку, QGIS и экспорт. Для multiclass формируется `int64` mask со значениями `-1/0/1/2`; `-1` игнорируется, `0` — общий фон, `1/2` — object types. Схема читается из manifest и строго сверяется со всеми GeoJSON.

Training UI автоматически выбирает `task=multiclass`, три выходных канала, `cross_entropy` или `cross_entropy_dice` и class-balanced sampling. Основной score — macro pixel F1 по object types. Также сохраняются per-class precision/recall/F1/IoU, micro F1 и foreground-vs-background F1. Общий threshold выбирается по максимальному macro F1, затем macro precision и большему threshold; threshold и schema сохраняются в checkpoint и MLflow.

Нативный inference использует softmax, argmax и общий threshold. Конфликты окон/сцен разрешаются по confidence, затем priority и class ID; морфология, векторизация и merge выполняются отдельно по типам. Канонический FeatureCollection содержит родительский `class_id` и `object_type_id/slug/name/color`, а ZIP — отдельные GeoJSON по типам.

В редакторе binary-режим не изменён. Для multiclass доступен единый segmented control «тип 1 / тип 2 / hard negative», переклассификация выбранного объекта, смысловые цвета, голубой контур выбранного объекта, легенда и counts. DTO/OpenAPI содержат `task`, `object_types`, `combined`, `source_status`, `source_changes` и class counts. Реализованы preview-token, `merge|replace`, проверка source/target snapshot с `409`, source staleness и атомарная Git-публикация.

В БД добавлены task, class schema и структурированные per-class metrics для training results, test samples, test tiles и test metrics; прежние скалярные поля сохранены для binary-совместимости.

## CUDA smoke семи архитектур

Проверка выполнена на RTX 5090 на реальном batch из двух тайлов 512×512 набора «Переувлажнения и заболачивания». В masks присутствовали `0/1/2`. Каждая сеть выполнила forward, `cross_entropy_dice`, backward и AdamW optimizer step; выход имел форму `[2, 3, 512, 512]`.

| Архитектура | Параметров | Loss | Peak CUDA bytes | Итог |
| --- | ---: | ---: | ---: | --- |
| `smp_deeplabv3plus_resnet50` | 26 681 235 | 2.3041956425 | 1 465 814 528 | ok |
| `smp_segformer_b2` | 24 726 019 | 1.9000904560 | 2 066 136 576 | ok |
| `smp_segformer_b3` | 44 601 859 | 1.7432404757 | 2 903 294 464 | ok |
| `smp_unet_resnet34` | 24 439 795 | 2.2057275772 | 1 230 416 384 | ok |
| `smp_unet_resnet50` | 32 524 531 | 2.3057208061 | 1 793 741 824 | ok |
| `smp_unet_resnet101` | 51 516 659 | 2.3346564769 | 2 329 329 152 | ok |
| `smp_unet_resnet152` | 67 160 307 | 1.8541598320 | 3 002 123 264 | ok |

Машинный отчёт сохранён на сервере в `architecture-smoke-batch2.json` внутри каталога релиза.

## Целевое обучение: ровно 10 эпох

- Job ID: `28ecc7e9-6262-402d-a1fa-3f24b159eb89`.
- TrainingResult ID: `33daefcd-a561-4d83-9d77-ba604d44e708`.
- MLflow experiment: `59`.
- MLflow run ID: `e5f8cd7f9fda46e7bf45a67e02c5e34e`.
- Статус: `completed`, MLflow `FINISHED`, выполнено 10/10 эпох.
- Время train loop: 76.243 s; полное время UI-job: 110.660 s.
- Checkpoints: `best.pt` — 99 015 777 bytes; `final.pt` — 99 016 131 bytes.

Конфигурация полностью соответствует заданию: `smp_segformer_b2`, tile 512, stride 256, augmentation 1, batch 4, class balance, factors `0.8/0/0.2`, `cross_entropy_dice`, LR `1e-5`, weight decay `1e-4`, до 72 train и 1000 val batches, patience 11, без временного лимита.

Train loss снизился с `1.9672709` до `1.2645808`; final val loss — `1.3115198`. Лучший checkpoint выбран на MLflow step 9:

| Метрика | Значение |
| --- | ---: |
| Confidence threshold | 0.3 |
| Macro pixel precision | 0.0487590826 |
| Macro pixel recall | 0.1783407581 |
| Macro pixel F1 | 0.0765806945 |
| Macro pixel IoU | 0.0414658292 |
| Micro pixel F1 | 0.1316878845 |
| Foreground-vs-background F1 | 0.1672869014 |
| `flooding` P/R/F1/IoU | 0 / 0 / 0 / 0 |
| `waterlogging` P/R/F1/IoU | 0.0975182 / 0.3566815 / 0.1531614 / 0.0829317 |

Два preflight-запуска выявили и закрыли реальные ошибки до финального обучения: job `531efe…` обнаружил отсутствие manifest в allowlist MLflow (`d3c95fa`), job `d022…` — попытку логировать отсутствующие object metrics (`a1a31d5`).

## Тестовая разметка и F1

TestSample `4f71217e-6ec4-41c4-91eb-d5a90ed14e99` содержит четыре тайла 512×512 и ровно 40 объектов: 20 `flooding` + 20 `waterlogging`, без предупреждения о дефиците. Для каждого тайла сохранены TIFF, class-ID PNG mask, цветной preview и GeoJSON. Уникальные значения masks по тайлам: `[0,1]`, `[0,1]`, `[0,1,2]`, `[0,2]`.

Структурированная оценка, записанная непосредственно в TrainingResult:

| Уровень | Результат |
| --- | --- |
| Pixel foreground | P `0.4240041`, R `0.0902280`, F1 `0.1487929` |
| Pixel typed | macro F1 `0.0798362`, micro F1 `0.1245205`; flooding `0`, waterlogging `0.1596724` |
| Objects, IoU ≥ 0.5 | macro/micro/foreground F1 `0`; TP 0, FP 44, FN 39 |

Дополнительная оценка той же test-разметки наложением финального ordinary pseudolabel результата:

| Уровень | Результат |
| --- | --- |
| Pixel foreground | P `0.6432101`, R `0.2681885`, F1 `0.3785425`; TP 44 386, FP 24 621, FN 121 117 |
| Pixel typed | macro F1 `0.2095571`, micro F1 `0.3401561`; flooding `0`, waterlogging `0.4191142` |
| Objects, IoU ≥ 0.5 | P `0.0434783`, R `0.05`, F1 `0.0465116`; TP 2, FP 44, FN 38 |
| Objects typed | macro F1 `0.0303030`, micro F1 `0.0465116`; flooding `0`, waterlogging `0.0606061` |

Это два отдельных поддерживаемых evaluation path: первая запись создаётся тестовым заданием TrainingResult, вторая — пространственным наложением сохранённого полного pseudolabel FeatureCollection. Неверный тип учитывается как FP предсказанного и FN истинного типа; object matching один-к-одному при IoU ≥ 0.5.

## Обычная и AOI-псевдоразметка

Финальный обычный job `51d765cb-da10-4b35-90cc-0275e54c32ef`, result `281beef6-5042-4770-8cd1-8560aca12ed5`:

- обработано 16/16 сцен, failed 0, missing 0;
- runtime 284.635 s;
- 28 324 объекта до merge, 27 874 после merge/conflict resolution;
- 22 `flooding` и 27 852 `waterlogging`;
- канонический GeoJSON — 44 124 409 bytes;
- ZIP по типам — 6 754 275 bytes, файлы `flooding.geojson` и `waterlogging.geojson`;
- schema и threshold `0.3` сохранены; у всех объектов присутствуют `class_id` и полный набор `object_type_*`.

На реальном результате из 28 тысяч объектов прежнее попарное разрешение конфликтов имело квадратичную сложность. Диагностический job был остановлен после более чем 23 минут postprocessing. После `da1219f` тот же этап со spatial index дал: load 0.216 s, merge 4.673 s, conflict resolution 2.059 s, при неизменном результате 27 874 объектов.

Финальный AOI job `24ab1b8d-5b9c-48c8-b7c5-c4d80160fbc3` завершился успешно:

- одна найденная и обработанная сцена, coverage 100%, warnings 0;
- 16 тайлов, 204 объекта до AOI merge, 30 после него;
- worker elapsed 3.317 s, полное время задания около 5.5 s;
- результат содержит 30 `waterlogging`; выбранная малая AOI не дала `flooding`, что допустимо условиями приёмки;
- `task=multiclass`, обе записи schema, threshold `0.3`, уникальные `candidate_id`, родительский `class_id`, confidence и полный `object_type_*` присутствуют у всех 30 объектов.

## QGIS

QGIS хранит одну каноническую review-сессию и общую очередь. Реализованы категоризированный слой, группа из двух синхронизированных представлений, фильтр object type и единые статусы проверки.

Поскольку малая AOI предсказала только `waterlogging`, оба UI-типа проверены на поднаборе из 52 реальных предсказаний ordinary job того же checkpoint: 22 `flooding` + 30 `waterlogging`, нормализованных к AOI review-контракту. QGIS `4.0.0-Norrköping` подтвердил:

- категоризированные цвета `#3B82F6` и `#22C55E`;
- group mode: 22/30 объектов в двух представлениях;
- фильтр `flooding`: 22/0 и 22 объекта в общей очереди;
- после принятия одного объекта: 21/0 во всех представлениях и 51 pending в общей очереди без фильтра;
- возврат в categorized mode удаляет временные views и восстанавливает канонический слой.

Полный автоматический QGIS suite также прошёл: 15/15.

## Triton и Geoalert

Экспорт TrainingResult создал архив `mlsystem2_flooding_waterlogging_b2_10ep_20260812_export.zip` размером 91 823 981 bytes. Metadata содержит `task=multiclass`, полную schema, threshold `0.3`, checkpoint metadata и ONNX opset 17.

Binary ABI не изменён. Multiclass-модель `mlsystem2_flooding_waterlogging_b2_10ep_20260812` имеет output `mask TYPE_UINT8`, динамическую форму `[B,2,H,W]`; два канала — thresholded one-hot foreground classes. Triton после загрузки вернул READY и metadata input `[1,4,-1,-1]`, output `[-1,2,-1,-1]`.

Прямой HTTP inference вернул `uint8 [1,2,512,512]`, только значения `0/1`, channel sums `[157,0]`, overlap между каналами `0`. Geoalert Compose на реальном test tile завершился за 0.399 s и создал оба требуемых выхода: `flooding.geojson` — 0 объектов, `waterlogging.geojson` — 24 объекта. После smoke модель штатно выгружена из Triton; repository state — `UNAVAILABLE`, reason `unloaded`.

## Автоматические проверки

- полный локальный pytest после performance fix: `430 passed, 10 skipped`;
- после финального изменения `class_id`: 40 целевых тестов и полный GitHub Actions workflow `31599925878`, success;
- frontend tests, TypeScript/OpenAPI public-contract checks и production build — success;
- QGIS 4 suite — 15/15, дополнительный real-output smoke — success;
- DB migration — `20260812_0016 (head)`;
- manifest/schema, conversion/priority/invalid geometry, merge/replace/409 conflicts, masks/class balance, metrics/threshold, test sample, pseudolabel, QGIS contracts, Triton export и ONNX parity покрыты тестами;
- семь реальных CUDA forward/backward/optimizer smoke — success.

## Итог приёмки

Полный технический цикл завершён. Новые датасеты доступны в live MLMarkup, MLSystem2 читает и редактирует multiclass без изменения binary/legacy поведения, обучение всех доступных архитектур запускается, целевой 10-эпоховый job завершён с конечными loss/metrics и рабочими checkpoints, а test, ordinary/AOI pseudolabel, оба QGIS-режима и Triton/Geoalert отработали на том же checkpoint.
