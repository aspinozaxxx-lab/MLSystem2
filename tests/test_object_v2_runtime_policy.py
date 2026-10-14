"""Точность и серверный параллелизм v2 не меняют исторические профили."""

from types import SimpleNamespace

import pytest

from mlsystem2.training_ui_api import _pseudo_runner, _worker
from mlsystem2.training_ui_api._inference_backend import GEOALERT_INFERENCE_BACKEND


@pytest.mark.parametrize("variant", [None, "legacy", "next_gen", "next_gen2", "object_f1", "object_f1_v2"])
def test_native_precision_is_full_fp32_only_for_v2(variant):
    matmul = SimpleNamespace(allow_tf32=True)
    cudnn = SimpleNamespace(allow_tf32=True)
    torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn))
    config = {"pipeline_variant": variant}

    _pseudo_runner._apply_native_object_precision(torch, config)

    assert matmul.allow_tf32 is (variant != "object_f1_v2")
    assert cudnn.allow_tf32 is (variant != "object_f1_v2")
    assert config == {"pipeline_variant": variant}


@pytest.mark.parametrize("variant", [None, "legacy", "next_gen", "next_gen2", "object_f1", "object_f1_v2"])
@pytest.mark.parametrize("backend", [GEOALERT_INFERENCE_BACKEND, "pytorch_one_off"])
def test_single_window_parallel_policy_is_only_v2_ortho(variant, backend):
    policy = _worker._object_v2_ortho_runtime_config(variant, backend)

    if variant == "object_f1_v2" and backend == GEOALERT_INFERENCE_BACKEND:
        assert policy == {
            "batch_size": 1,
            "batch_policy": "geoalert_single_window_default_v1",
            "scene_workers": 2,
            "inference_precision": "fp32_no_tf32",
        }
    else:
        assert policy == {}
