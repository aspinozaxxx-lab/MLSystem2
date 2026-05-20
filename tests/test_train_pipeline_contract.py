from __future__ import annotations

import inspect
from typing import get_type_hints

from mlsystem2.train_pipeline.api import run_train_pipeline
from mlsystem2.train_pipeline.contracts import TrainPipelineRequest, TrainPipelineResult


def test_run_train_pipeline_signature_uses_request_contract() -> None:
    signature = inspect.signature(run_train_pipeline)
    parameters = list(signature.parameters.values())
    hints = get_type_hints(run_train_pipeline)
    assert len(parameters) == 1
    assert parameters[0].name == "request"
    assert hints["request"] is TrainPipelineRequest


def test_train_pipeline_result_fields() -> None:
    assert {"status", "mlflow_run", "timings", "report"} <= set(TrainPipelineResult.model_fields)
