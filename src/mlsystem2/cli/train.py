"""Точка входа обучения из командной строки."""

from __future__ import annotations

import argparse
import signal

from mlsystem2.settings.api import load_settings
from mlsystem2.train_pipeline.api import run_train_pipeline
from mlsystem2.train_pipeline.contracts import PipelineStatus, TrainPipelineRequest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-train")
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу.")
    parser.add_argument("--run-name", default=None, help="Необязательное имя запуска.")
    args = parser.parse_args(argv)

    _install_signal_handlers()
    load_settings(args.config)
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
