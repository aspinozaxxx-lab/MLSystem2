# Альтернативный конвейер `next-gen` и SegFormer B0 для `Заболачивание/main`

Дата: 2026-09-05.

## Итог

В MLSystem2 реализован альтернативный конвейер обучения `next-gen`. Он включается полем
`train.pipeline_variant=next_gen` в шаблоне или ручной конфигурации. Конфигурации и checkpoint без этого
поля по-прежнему считаются `legacy`; существующий путь обучения не изменён и остаётся воспроизводимым.

Для `Заболачивание/main` по screening выбран следующий вариант:

- архитектура `segformer_b0` — отдельный HF SegFormer B0;
- pretrained-веса `nvidia/segformer-b0-finetuned-ade-512-512` с закреплённой revision
  `489d5cd81a0b59fab9b7ea758d3548ebe99677da`;
- нормализация `imagenet_rgb_red_nir`;
- loss `focal_tversky`;
- sampling `positive/background=0,4/0,6`;
- tile/context/stride `512/128/256`, batch `8`, augmentation level `2`;
- AdamW, LR `6e-5`, weight decay `0,01`, полный validation каждые 5 эпох;
- оптимизация порога по validation и production merge `core_crop`.

На честном цикле из шести held-out сцен результат HF-варианта составил:

- macro mean pixel F1: `0,15218`;
- population std: `0,10644`;
- min: `0,03044`;
- global micro pixel F1: `0,20388`;
- fixed-0,5 macro pixel F1: `0,11018`;
- macro object F1: `0,02602`.

Разброс между сценами большой, а object F1 низкий. Реализация конвейера технически проверена, но эти
метрики не дают оснований автоматически назначать модель основной. Внешней независимой территории в
испытании не было.

## Что отличалось в NewPipeline

В `D:\Projects\NewPipeline` результат получался следующим путём:

- HF checkpoint `nvidia/segformer-b0-finetuned-ade-512-512`;
- patch `512`, stride `256`, batch `8`, до 50 эпох, LR `1e-4`;
- четвёртый канал первого convolution инициализировался копией RED;
- weighted sampler с весом positive `15`;
- CrossEntropy с весами классов;
- AdamW и `ReduceLROnPlateau` по validation loss, early stopping по validation loss;
- Gaussian merge с `sigma=patch_size/4`;
- сохранялся только `state_dict`.

Одновременно в NewPipeline были риски, которые нельзя переносить буквально:

- train/validation/test делились случайно по перекрывающимся окнам одной территории, поэтому возможна
  пространственная утечка;
- применялась per-window per-channel min-max нормализация, из-за чего одинаковое значение пикселя меняло
  смысл от окна к окну;
- `ignore_mismatched_sizes=True` скрывал несовместимости head при загрузке;
- использовался двухклассовый head вместо одного binary logit;
- checkpoint не содержал полной конфигурации модели, preprocessing и provenance;
- Gaussian-обход `range(0, size-patch+1, stride)` мог не покрывать правую и нижнюю границы;
- nodata не исключался единым valid-mask из loss и метрик.

## Сравнение конвейеров

| Область | `legacy` MLSystem2 | NewPipeline | `next-gen` MLSystem2 |
|---|---|---|---|
| Переключение | default, старые YAML/checkpoint | отдельный экспериментальный код | `train.pipeline_variant=next_gen` в тех же публичных фасадах |
| Split | случайные окна, balanced validation | случайные окна train/val/test | детерминированный scene-fold и spatial purge |
| Validation | balanced subset, optional limit | оконный subset | все valid-окна held-out сцен |
| Nodata/padding | прежняя трактовка как background | не единообразно исключены | valid-mask исключает из loss, TP/FP/FN и фотометрии |
| Нормализация | прежняя | min-max каждого окна | один воспроизводимый профиль внутри model wrapper |
| HF-модель | прежний путь | pretrained, два logits | pinned pretrained, 4 канала, один binary logit, offline restore |
| Scheduler | cosine по эпохам | plateau по val loss | plateau по полной scene-macro pixel F1 |
| Порог | прежняя сетка/фиксированный | фиксированный | fixed либо 4096-bin streaming optimization |
| Повторный draw | прежняя генерация | зависит от sampler | seed зависит от epoch, draw, scene и window |
| Артефакты | прежний набор | в основном `state_dict` | resolved config, split, preprocessing, runtime, scene metrics, hashes |
| Merge | production core-crop | Gaussian | core-crop; Gaussian только финальный A/B |

Из NewPipeline заимствованы удачные исходные идеи — HF pretrained B0, адаптация `3→4` каналов, геометрия
`512/256`, AdamW, plateau scheduler и Gaussian-диагностика. Они встроены в существующие архитектурные
границы MLSystem2 и дополнены защитой от утечек, valid-mask, полным checkpoint и воспроизводимыми
артефактами.

## Реализация

Основная реализация выполнена без нового верхнеуровневого модуля:

- `settings` валидирует совместимость variant, binary/каналов/архитектуры и запрещает ограниченный val;
- `dataset_preparing` строго проверяет `uint8`, четыре канала, CRS и band contract
  `RED, GRN, BLU, NIR`;
- `tile_preparation` реализует scene-fold, spatial purge, valid-mask, epoch/draw-aware sampler и совместные
  геометрические аугментации image/target/mask;
- `models` хранит preprocessing в wrapper, поддерживает три профиля и полностью offline HF restore;
- `train` выполняет полную периодическую validation, `ReduceLROnPlateau`, early stopping по validation
  events и потоковую оптимизацию threshold;
- `train_pipeline` формирует split/preprocessing/runtime/scene/Gaussian отчёты;
- `mlflow_adapter` безопасно flatten-логирует параметры и сохраняет hashes/provenance;
- `training_ui_api` и UI показывают variant/fold, отдельную HF B0 архитектуру и допустимые поля.

Во время серверной проверки дополнительно найдены и исправлены две ошибки:

1. HF ONNX экспортировал output с лишним channel dimension для strict Triton-контракта.
2. Gaussian A/B неверно срезал valid-mask формы `[B,H,W]` как `[B,1,H,W]`.

Также production-контейнер без `.git` первоначально не мог записать реальный code commit. Теперь ревизия
берётся из проверенного `/opt/mlsystem2/repo/DEPLOYED_COMMIT`, а источник сохраняется в runtime report.

Коммиты реализации:

| Commit | Назначение |
|---|---|
| `7a695496dcbf598cddd831ebc7426ece7df0c58e` | основной `next-gen` конвейер, UI, тесты и документация |
| `308891574af89adc592446759a821689a0c8029b` | strict HF ONNX/Triton output contract |
| `76ac787a6bfe7cca9d9e7a42bf31185cceb54c1e` | valid-mask в Gaussian A/B |
| `501042f45301a5e2c7e32ff1ab5881c755f2561d` | точная production code revision и provenance |

## Датасет и схема fold

Все сравниваемые `next-gen` задания привязаны к одной ревизии:

- dataset key: `6decf79c-7f08-4b88-9566-bb97d760936a`;
- UI version: `managed:2:git:c9dbc74b15c7272b29aacd99c3e44e4a90dc4690`;
- dataset revision SHA-256:
  `5cd96aecd5233083d1251a97fb1e5ea8b55a38206d0e677b609b7e1ef35df6c9`.

При seed `42`, `val_fraction=0,2` и шести сценах каждый fold удерживает одну сцену:

| Fold | Held-out сцена | Purged train windows |
|---:|---|---:|
| 0 | `KV3_30937_31845-00_KANOPUS_20230831_034637_15.L2.PMS.SCN04` | 154 |
| 1 | `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07` | 164 |
| 2 | `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN05` | 0 |
| 3 | `KV3_30937_31845-00_KANOPUS_20230831_034637_15.L2.PMS.SCN03` | 170 |
| 4 | `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN08` | 153 |
| 5 | `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN07` | 0 |

Split manifest подтвердил нулевое географическое пересечение train-входов с расширенным valid footprint
held-out сцены. Purge сработал в том числе для перекрывающихся пар `SCN07↔SCN08` и `SCN03↔SCN04`.

## Smoke

Три обязательных одноэпоховых smoke завершились успешно:

| Вариант | MLflow run ID | Результат аудита |
|---|---|---|
| legacy SMP B0 | `9ee1fef3fc2a46cc854aabe9b57d03c5` | passed |
| next-gen SMP B0 | `dc2fddcea91f471586466e0850953f41` | passed |
| next-gen HF B0 pretrained | `7d1db1787bd841d9a742020f0a65142d` | passed |

## Screening на SCN03 и подтверждение на SCN07

Все screening jobs: 25 эпох, полная validation каждые 5 эпох, лимит 1200 секунд. Основной критерий —
scene pixel F1.

| Вариант | Pretrained | Нормализация | Loss | Positive | Fold | Best F1 | MLflow run ID |
|---|---:|---|---|---:|---:|---:|---|
| normalization | да | `scale_255` | BCE+Dice | 0,5 | 3 | 0,30767 | `58e80ea394d948cdaf7e808df27e816d` |
| normalization | да | `imagenet_rgb_red_nir` | BCE+Dice | 0,5 | 3 | 0,33172 | `fcf2451e7c274151805a215b62b02a48` |
| normalization | да | `robust_percentile` | BCE+Dice | 0,5 | 3 | 0,30225 | `9ce7af483cbd4fe686b0495f876e7a7b` |
| pretrained A/B | нет | `imagenet_rgb_red_nir` | BCE+Dice | 0,5 | 3 | 0,30810 | `1cba5b75b7e746b799cf99f025cacfb5` |
| loss A/B | да | `imagenet_rgb_red_nir` | focal-Tversky | 0,5 | 3 | 0,35193 | `26bf77032f6248cdb6e74a5b7e7c344e` |
| sampling | да | `imagenet_rgb_red_nir` | focal-Tversky | 0,4 | 3 | **0,38459** | `8fcc9f99a5964da783d30e8f40b74540` |
| sampling | да | `imagenet_rgb_red_nir` | focal-Tversky | 0,6 | 3 | 0,32988 | `b69e24b9e00a4b33819eab396dfe7993` |
| подтверждение победителя | да | `imagenet_rgb_red_nir` | focal-Tversky | 0,4 | 1 | 0,10659 | `9da460bfaad94e82bb26cb867002b478` |

На SCN03 pretrained дал `+0,02362 F1` относительно случайной инициализации при BCE+Dice. ImageNet
нормализация победила `scale_255` и `robust_percentile`; focal-Tversky и positive `0,4` дали лучший
screening. Резкое снижение на SCN07 подтвердило необходимость полного цикла по всем сценам.

## Финальная шестисценовая оценка HF B0

Все шесть запусков использовали commit `501042f45301a5e2c7e32ff1ab5881c755f2561d`, одну dataset revision,
до 60 эпох и лимит 3300 секунд.

| Fold | Best epoch | Threshold | Pixel F1 | Precision | Recall | Fixed 0,5 F1 | Object F1 | Время, с | MLflow run ID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 15 | 0,18339 | 0,14410 | 0,11426 | 0,19502 | 0,12833 | 0,03320 | 341,6 | `446393d80fe14604be1c95ce933b7b5c` |
| 1 | 10 | 0,02564 | 0,12556 | 0,08427 | 0,24618 | 0,01101 | 0,01678 | 294,8 | `3b31da4abd2a4401a075d77b73d38ef9` |
| 2 | 1 | 0,38950 | 0,11893 | 0,06740 | 0,50523 | 0,05720 | 0,01531 | 197,7 | `ac4d144359f441c18224e2321b5ba135` |
| 3 | 15 | 0,03028 | 0,37604 | 0,34623 | 0,41146 | 0,31976 | 0,05685 | 346,8 | `641e645493a045ab9f301054d4058e5c` |
| 4 | 45 | 0,38510 | 0,11800 | 0,08759 | 0,18075 | 0,11453 | 0,01575 | 609,0 | `e095427a6e8b4909a9345f05d440c61d` |
| 5 | 45 | 0,43248 | 0,03044 | 0,01575 | 0,45745 | 0,03026 | 0,01824 | 580,6 | `af48f17866a044e28c019c0e84cc5a35` |

Агрегаты:

| Метрика | Значение |
|---|---:|
| Macro mean pixel F1 | 0,15218 |
| Macro pixel F1 population std | 0,10644 |
| Min scene pixel F1 | 0,03044 |
| Macro precision / recall | 0,11925 / 0,33268 |
| Global micro F1 / precision / recall | 0,20388 / 0,13481 / 0,41807 |
| Fixed-0,5 macro F1 | 0,11018 |
| Fixed-0,5 global micro F1 | 0,17707 |
| Macro object F1 / precision / recall | 0,02602 / 0,01428 / 0,18194 |
| Максимальная peak VRAM | 3 046 015 488 байт |
| Суммарное время обучения | 2370,5 с / 0,6585 GPU-часа |

## Контрольные SMP и legacy обучения

`next-gen SMP B0` прогнан на тех же folds 1 и 3 с конфигурацией победителя, кроме архитектуры и
`pretrained=false`:

| Вариант | Fold / область оценки | Best epoch | Threshold | Pixel F1 | Precision | Recall | Fixed 0,5 F1 | Object F1 | Время, с | MLflow run ID |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| next-gen SMP B0 | fold 1 / held-out SCN07 | 10 | 0,26911 | 0,18897 | 0,14308 | 0,27821 | 0,16138 | 0,01986 | 296,1 | `7f727adeb8104ef69153517e04008f3b` |
| next-gen SMP B0 | fold 3 / held-out SCN03 | 50 | 0,75629 | 0,40755 | 0,31546 | 0,57559 | 0,39972 | 0,08853 | 579,4 | `6507c204dd1d4ae6b1784fefa601a5ad` |
| legacy SMP B0 | balanced window validation | 105 | 0,75 | 0,70907 | 0,66747 | 0,75620 | — | 0,21557 | 1017,3 | `c760ef7137f54fe39d56d46751635e76` |

На двух контрольных сценах SMP превысил HF на `0,06341 F1` для fold 1 и на `0,03151 F1` для fold 3.
Это основание прогнать SMP по всем шести folds в следующей серии, но не основание считать его лучше по
всему датасету: сейчас у SMP есть только две независимые точки, а у HF — полный шестисценовый цикл.

Legacy запуск использовал текущий шаблон буквально без изменения: tile/context/stride `768/256/364`,
batch `4`, augmentation `3`, sampling `0,8/0,2`, `1000` эпох, LR `1e-5`, cosine scheduler, patience `20`,
ограничение `72/1000` train/val batch и 3600 секунд. Он завершился early stopping на эпохе 125. Его
`0,70907 F1` нельзя сравнивать с held-out-scene результатами: balanced оконная validation содержит окна
тех же сцен, что и train, и сохранена только для проверки воспроизводимости старого пути.

## Gaussian A/B

Gaussian применялся только как финальная диагностика best checkpoint. На всех сценах подтверждены:

- полное покрытие valid footprint, `uncovered_valid_pixels=0`;
- отсутствие positive prediction на nodata;
- сохранение production merge `core_crop` для checkpoint metadata, ONNX и Geoalert.

Средний Gaussian pixel F1 равен `0,04594` против `0,15218` у core-crop. Gaussian оказался немного лучше
только на fold 2 (`0,12304` против `0,11893`) и значительно хуже в остальных folds. Поэтому включать его
в production или в шаблон по умолчанию не следует.

## Аудит артефактов

Для каждого успешного smoke, confirmation и каждого из шести финальных folds выполнены:

- загрузка `best.pt` на CPU с `HF_HUB_OFFLINE=1`;
- проверка variant, preprocessing, band contract, split/fold, scheduler, threshold policy, pinned HF
  provenance, dataset revision и code commit;
- сверка SHA-256 checkpoint с MLflow hash report;
- checkpoint round-trip: max abs logits `0`;
- ONNX export и `onnx.checker`;
- PyTorch/ONNX/Triton raw-logit parity;
- одинаковая binary mask на реальном контрольном tile;
- сохранение границ выходного изображения;
- пустой прогноз на полностью nodata tile;
- загрузка, GPU inference и выгрузка временной Triton-модели.

Максимальное расхождение raw logits PyTorch↔Triton среди шести folds — `5,24521e-6`, что лучше лимита
`1e-4`. На каждом реальном tile расхождение binary mask равно `0` пикселей. Все временные модели после
проверки выгружены.

Контрольные SMP/legacy checkpoint также прошли offline load, точный round-trip, ONNX checker, GPU Triton,
проверку границ и safe-control mask. Максимальное расхождение logits равно `7,62939e-6`; safe-control
masks совпали побитово. На дополнительном реальном tile из 16 384 пикселей бинаризация разошлась на
1 пиксель для SMP fold 1 и на 2 пикселя для SMP fold 3/legacy: соответствующие logits находились на самой
границе threshold, при соблюдении допуска raw logits. Для `next-gen` nodata prediction остался пустым.
Legacy дал 2186 positive пикселей на нулевом входе, что зафиксировано как прежнее поведение, а не исправлено
задним числом. У legacy также ожидаемо нет нового `reports/checkpoint_hashes.json`; SHA-256 самого файла
вычислен аудитом, но старый набор MLflow-артефактов не расширялся.

SHA-256 `best.pt`:

| Fold | SHA-256 |
|---:|---|
| 0 | `7b9e3a35de94379f13e876549ceb234888e60913c453fafb4fd8b47faeb32f4d` |
| 1 | `c67707bab514f4957706c9ed090f267593e30dd8666ad941e5bc4896a4fe4294` |
| 2 | `9bfc782d09e488d157a3ec3aa7fba504db75f43498903cb5edf1ab75158157c4` |
| 3 | `075a324c0aa7955a62743aa9831eb25a9e1c7be581dfaf0a6966e173e70bd422` |
| 4 | `273ce8344a30068955afc3b93e829a5a62d86f1bff400dfc4242469560d23a2b` |
| 5 | `f617d5e1f1b60de7fb26b26c070f43b0471a4eab75157258745a53a91aea9f86` |

## Проверки кода и развёртывания

Локально на commit `501042f45301a5e2c7e32ff1ab5881c755f2561d`:

- Python: `571 passed, 2 skipped`;
- Ruff для `src` и `tests`: passed;
- frontend typecheck: passed;
- frontend tests: `49 passed`;
- frontend production build: passed;
- OpenAPI snapshot/contract: passed;
- architecture boundaries и public facades: passed;
- synthetic tiny-overfit отдельно для SMP B0 и HF B0: passed.

CI run `33933044865` завершён успешно. После deployment серверный `DEPLOYED_COMMIT`, runtime report и
checkpoint metadata совпали с `501042f45301a5e2c7e32ff1ab5881c755f2561d`; health API, worker и GPU
проверены.

## Dataset-template победителя

После завершения всех метрик и аудитов создан отдельный шаблон:

- id: `839c5b9f-1e44-4445-9730-db8fe595a39e`;
- имя: `SegFormer B0 HF (next-gen) / Заболачивание\main`;
- parent: `c19f8357-40c6-48eb-a063-660902cfdc71`;
- source: `manual`, version `2`, active `true`;
- fold по умолчанию: `0`; для полного цикла его нужно менять явно;
- Gaussian A/B по умолчанию: `false`.

Публичный HTTP API после commit транзакции вернул точные winner-параметры. Для пары
`Заболачивание/main + segformer_b0` правило автоматики осталось `id=null`, `training_enabled=false`,
`pseudo_markup_enabled=false`; глобальная автоматика сохранила прежнее состояние. Основная сеть класса не
изменилась: результат `e85f1801-e2fd-4d6b-9db8-40af577a4c6b` по-прежнему имеет `is_primary=true` на
исходном датасете `d63909cc-cf0a-4d82-ae6e-e95adea790f7`.

## Диагностические запуски, исключённые из оценки

Из метрик выше исключены:

- `bf463a459737495f96a3b177374aadb2` — запуск до исправления Gaussian valid-mask;
- `8f0459c3...` — остановленный старый fold 1;
- `13d8bcc300674cce856a67c16bc4ccc8` — отменённый запуск до исправления production provenance;
- `c6866eff...` — остановленный запуск с недопустимой sampling-конфигурацией.

Они сохранены только как диагностическая история и не участвовали в выборе параметров или агрегатах.

## Ограничения и решение по эксплуатации

- Шесть folds покрывают все сцены текущего датасета, но не внешнюю независимую территорию.
- Низкий minimum и большой std показывают слабую переносимость между сценами.
- Оптимальные thresholds сильно различаются (`0,02564…0,43248`), что указывает на нестабильную
  калибровку вероятностей.
- Legacy balanced window validation и существующий test sample не являются независимой оценкой и должны
  использоваться только диагностически.
- Gaussian merge не показал преимущества и остаётся выключенным.
- Автоматическое правило обучения не создавалось; ни один результат основной сетью автоматически не
  назначался.

Технически `next-gen` готов для контролируемых сравнительных обучений. Для решения о production-модели
нужна разметка независимой территории и улучшение устойчивости на folds 1, 2, 4 и особенно 5.
