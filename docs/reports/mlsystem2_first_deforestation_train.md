# Первый запуск обучения MLSystem2 по вырубкам

## Сервер
- адрес: `root@31.192.104.147`, имя хоста `70648.example.ru`
- путь репозитория: `/data/MLSystem2`
- коммит: `1c614b5`
- python: `Python 3.12.3`, `/usr/bin/python3`
- torch: сначала `2.12.0+cu130`, затем проверочно переустановлен `2.11.0+cu128`
- cuda: недоступна для PyTorch, `torch.cuda.is_available() == False`
- gpu: `lspci` видит NVIDIA `10de:2b85`, но `nvidia-smi` падает с `Driver/library version mismatch`

## Датасет
- каталог снимков: `/data/mlsystem2/prepared_images/kanopus`
- подготовленные снимки: 995 GeoTIFF
- список сцен: `/data/mlsystem2/first_train/mlmarkup_deforestation/deforestation.txt`
- файл разметки: `/data/mlsystem2/first_train/mlmarkup_deforestation/deforestation.geojson`
- символьная ссылка: `/data/mlsystem2/first_train/mlmarkup_deforestation -> /data/MLMarkup/Вырубки`
- отчет подготовки датасета: `ok`
- разбиение train/val: найдено 35/35 сцен, всего 532 объекта
- train: 29 сцен, 419 объектов
- val: 6 сцен, 113 объектов
- smoke train loader: 6072 batch при `batch_size=4`
- первый batch image: shape `(4, 4, 1024, 1024)`, dtype `torch.float32`, min `12.0`, max `255.0`
- первый batch mask: shape `(4, 1, 1024, 1024)`, dtype `torch.float32`, min `0.0`, max `0.0`

## Гиперпараметры
- источник: сильная 1024-ветка старого MLSystem/MLflow и параметры из задачи на запуск
- конфиг: `/data/mlsystem2/first_train/deforestation_first_1024.yaml`
- tile_size: 1024
- stride: 512
- batch_size: 4
- learning_rate: `0.000002`
- weight_decay: `0.0001`
- loss: `focal_tversky`
- threshold: `0.75`
- epochs: 10
- patience: 5
- initial_checkpoint_uri: `null`
- проверенный checkpoint-кандидат: `/data/mlsystem/airflow/status/manual_deforestation_20260514_002020_posweight_recall/segformer_b2.pt`
- результат загрузки checkpoint: файл читается как torch dict с `model_state_dict`, но веса несовместимы с текущими именами ключей Hugging Face `SegformerForSemanticSegmentation`, поэтому warm-start отключен.

## MLflow
- tracking_uri: `http://127.0.0.1:5000`
- experiment_name: `MLSystem2-First`
- experiment_id: `43`
- run_name: не создан
- run_id: не создан
- статус: train run не стартовал, потому что CUDA недоступна до входа в обучение
- наблюдаемые метрики: нет

## Лимит Времени
- запланированная команда: `timeout --preserve-status 3600s python -m mlsystem2.cli.train --config /data/mlsystem2/first_train/deforestation_first_1024.yaml --run-name <run_name>`
- timeout: 3600s
- фактическая длительность: обучение не стартовало
- timeout достигнут: нет

## Артефакты
- best_checkpoint_path: не создан
- final_checkpoint_path: не создан
- логи: `/data/mlsystem2/first_train/logs`
- серверный конфиг: `/data/mlsystem2/first_train/deforestation_first_1024.yaml`

## Проверки
- `python -m pytest tests/test_public_contracts.py`: пройдено, 1 тест
- `python -m pytest tests -q`: пройдено, 65 тестов
- `python -m ruff check src tests`: пройдено
- smoke dataset/tile: пройден, raw image values выше 1.0 подтверждены

## Проблемы И Исправления
- `pip install -e ".[torch,dev]"` установил `torch 2.12.0+cu130`, но CUDA осталась недоступна. Виртуальное окружение проверочно переустановлено на `torch 2.11.0+cu128` и `torchvision 0.26.0+cu128`; CUDA все равно осталась недоступна.
- `nvidia-smi` сообщает `Driver/library version mismatch`: загруженный kernel module имеет версию `580.142`, установленные userland-библиотеки и пакетный kernel module имеют версию `580.159.03`. PyTorch сообщает CUDA error 804 и не может перечислить устройства.
- Первая запись runtime-конфига через shell повредила кириллический путь `Вырубки` в `???????`. Исправлено созданием ASCII symlink `/data/mlsystem2/first_train/mlmarkup_deforestation`.
- Старые checkpoints по вырубкам несовместимы с текущей HF-реализацией SegFormer B2 по именам ключей state dict. Compatibility shim не добавлялся, потому что это несовместимость архитектуры и layout ключей, а не простой legacy envelope checkpoint.

## Блокер
Первый реальный запуск обучения MLSystem2 не стартовал, потому что GPU stack сервера неконсистентен до входа в train loop. Запуск часовой команды с `device: cuda` сейчас создаст failed MLflow run без эпох и полезных метрик. Следующее необходимое инфраструктурное действие: контролируемая перезагрузка NVIDIA driver или reboot сервера, чтобы загруженный kernel module и userland-библиотеки совпадали по версии `580.159.03`.
