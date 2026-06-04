"""Точка входа обучения из командной строки."""

from __future__ import annotations

import argparse
import os
import signal

from mlsystem2.settings.api import load_settings
from mlsystem2.train_pipeline.api import run_train_pipeline
from mlsystem2.train_pipeline.contracts import PipelineStatus, TrainPipelineRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-train")
    parser.add_argument("--config", default=None, help="Путь к legacy YAML-конфигу полного запуска.")
    parser.add_argument("--settings", default=None, help="Путь к settings.yml с параметрами приложения.")
    parser.add_argument("--run", default=None, help="Путь к run.yml с параметрами конкретного обучения.")
    parser.add_argument("--run-name", default=None, help="Необязательное имя запуска.")
    args = parser.parse_args(argv)

    _install_signal_handlers()
    if args.run:
        settings_path = args.settings or os.getenv("MLSYSTEM2_SETTINGS_PATH") or "configs/settings.server.yaml"
        load_settings(settings_path, args.run)
    elif args.config:
        load_settings(args.config)
    else:
        parser.error("нужно указать либо --config, либо --run с опциональным --settings")
    result = run_train_pipeline(TrainPipelineRequest(run_name=args.run_name))
    print(f"status={result.status.value}")
    if result.mlflow_run is not None:
        print(f"mlflow_run={result.mlflow_run.run_id}")
    return 0 if result.status == PipelineStatus.SUCCEEDED else 1


def _install_signal_handlers() -> None:
    def _raise_interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"Получен сигнал остановки: {signum}")

    signal.signal(signal.SIGTERM, _raise_interrupted)


if __name__ == "__main__":
    raise SystemExit(main())
