# Обновление компонентов

Дата: 2026-05-22.

## Установлено
- Python 3.12.10 установлен через:
  `winget install --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements --silent`

## Проверка
- `python --version` после добавления user Python в `PATH` текущей сессии возвращает `Python 3.12.10`.
- `python -m py_compile Geoalert\Workflow Engine\inference-v1.5.5\modules\urban\urban\bricks\adapters.py` проходит без ошибок.

## Остаточные блокеры
- Docker Desktop не установлен.
- WSL не установлен.
- Полный запуск inference worker требует приватный пакет `we-queue-client[minio]==1.5.2` из GitLab PyPI и восстановленные несекретные значения конфигурации вместо санитизированных фрагментов.
