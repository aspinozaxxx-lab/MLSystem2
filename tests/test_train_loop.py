from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from mlsystem2.models.contracts import ModelHandle, ModelSpec
from mlsystem2.tile_preparation.contracts import HARD_NEGATIVE_LABEL
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import EpochMetrics, TrainConfig, TrainError, TrainRequest


def test_train_model_smoke_saves_checkpoints(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=TinySegmentationModel(),
    )
    train_loader = _fake_loader(torch)
    val_loader = _fake_loader(torch)

    result = train_model(
        TrainRequest(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainConfig(
                epochs=1,
                batch_size=2,
                seed=7,
                inference_context=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            sample_size=16,
        )
    )

    assert result.epochs_total == 1
    assert len(result.history) == 1
    assert Path(result.best_checkpoint_path).is_file()
    assert Path(result.final_checkpoint_path).is_file()
    assert result.history[0].train_loss >= 0.0
    assert result.history[0].val_loss >= 0.0
    assert 0.0 <= result.history[0].val_best_threshold_pixel_f1 <= 1.0
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert checkpoint["metadata"]["sample_size"] == 16
    assert checkpoint["metadata"]["inference_context"] == 2
    assert checkpoint["metadata"]["inference_core_size"] == 12
    assert checkpoint["metadata"]["seed"] == 7


def test_train_model_multiclass_cross_entropy_smoke(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinyMulticlassModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 3, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=3),
        model=TinyMulticlassModel(),
    )

    result = train_model(
        TrainRequest(
            model=model,
            train_loader=_fake_multiclass_loader(torch),
            val_loader=_fake_multiclass_loader(torch),
            config=TrainConfig(
                task="multiclass",
                epochs=1,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="cross_entropy",
                threshold=0.5,
                early_stopping_patience=1,
                class_slugs=["class_a", "class_b"],
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    metrics = result.history[0]
    assert result.epochs_total == 1
    assert 0.0 <= metrics.val_best_threshold_pixel_f1 <= 1.0
    assert 0.0 <= metrics.val_best_threshold_precision <= 1.0
    assert 0.0 <= metrics.val_best_threshold_recall <= 1.0
    assert Path(result.best_checkpoint_path).is_file()
    assert Path(result.final_checkpoint_path).is_file()


def test_multiclass_cross_entropy_dice_loss_is_finite() -> None:
    torch = pytest.importorskip("torch")
    from mlsystem2.train import _trainer

    logits = torch.randn((2, 3, 8, 8), dtype=torch.float32)
    masks = torch.zeros((2, 8, 8), dtype=torch.long)
    masks[:, 2:5, 2:5] = 1
    masks[:, 5:7, 5:7] = 2
    config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=2,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="cross_entropy_dice",
        threshold=0.5,
        early_stopping_patience=1,
        class_slugs=["class_a", "class_b"],
    )

    loss = _trainer._loss(torch, logits, masks, config)

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_train_model_respects_batch_limits(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=TinySegmentationModel(),
    )
    result = train_model(
        TrainRequest(
            model=model,
            train_loader=_fake_loader(torch),
            val_loader=_fake_loader(torch),
            config=TrainConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
                max_train_batches_per_epoch=1,
                max_val_batches_per_epoch=1,
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert result.epochs_total == 1
    assert len(result.history) == 1


def test_train_model_stops_after_training_time_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    monkeypatch.setattr(_trainer, "_training_time_exceeded", lambda config, total_started: True)
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=TinySegmentationModel(),
    )

    result = train_model(
        TrainRequest(
            model=model,
            train_loader=_fake_loader(torch),
            val_loader=_fake_loader(torch),
            config=TrainConfig(
                epochs=3,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=3,
                max_training_time_sec=1,
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert result.epochs_total == 1
    assert Path(result.final_checkpoint_path).is_file()


def test_train_model_accepts_batch_metadata(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=TinySegmentationModel(),
    )
    result = train_model(
        TrainRequest(
            model=model,
            train_loader=_fake_loader(torch, with_meta=True),
            val_loader=_fake_loader(torch, with_meta=True),
            config=TrainConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert result.epochs_total == 1
    assert 0.0 <= result.history[0].val_best_threshold_pixel_f1 <= 1.0


def test_validation_pixel_f1_counts_known_confusion_matrix() -> None:
    torch = pytest.importorskip("torch")

    class IdentityModel(torch.nn.Module):
        def forward(self, images):
            return images

    from mlsystem2.train import _trainer

    logits = torch.tensor([[[[-10.0, 10.0], [10.0, -10.0]]]], dtype=torch.float32)
    masks = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=1,
    )

    result = _trainer._validate_epoch(
        torch,
        IdentityModel(),
        [(logits, masks, {"augmented_tile_count": 0})],
        torch.device("cpu"),
        config,
        1,
    )

    assert result["best_threshold"] == 0.3
    assert result["best_threshold_pixel_f1"] == 0.5
    assert result["best_threshold_precision"] == 0.5
    assert result["best_threshold_recall"] == 0.5


def test_validation_pixel_f1_counts_background_false_positive() -> None:
    torch = pytest.importorskip("torch")

    class IdentityModel(torch.nn.Module):
        def forward(self, images):
            return images

    from mlsystem2.train import _trainer

    logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]], dtype=torch.float32)
    masks = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=1,
    )

    result = _trainer._validate_epoch(
        torch,
        IdentityModel(),
        [(logits, masks)],
        torch.device("cpu"),
        config,
        1,
    )

    assert result["best_threshold_pixel_f1"] == pytest.approx(2.0 / 3.0)
    assert result["best_threshold_precision"] == 0.5
    assert result["best_threshold_recall"] == 1.0


def test_validation_pixel_f1_is_zero_without_gt_positives() -> None:
    torch = pytest.importorskip("torch")

    class IdentityModel(torch.nn.Module):
        def forward(self, images):
            return images

    from mlsystem2.train import _trainer

    logits = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    masks = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=1,
    )

    result = _trainer._validate_epoch(
        torch,
        IdentityModel(),
        [(logits, masks)],
        torch.device("cpu"),
        config,
        1,
    )

    assert result["best_threshold_pixel_f1"] == 0.0


def test_validation_object_f1_selects_object_threshold() -> None:
    torch = pytest.importorskip("torch")

    class IdentityModel(torch.nn.Module):
        def forward(self, images):
            return images

    from mlsystem2.train import _trainer

    probabilities = torch.tensor([[[[0.8, 0.8, 0.6, 0.6, 0.6]]]], dtype=torch.float32)
    logits = torch.logit(probabilities)
    instances = torch.tensor([[[1, 1, 0, 0, 0]]], dtype=torch.long)
    masks = (instances > 0).to(dtype=torch.float32).unsqueeze(1)
    config = TrainConfig(
        quality_metric="objects",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=1,
    )

    result = _trainer._validate_epoch(
        torch,
        IdentityModel(),
        [(logits, masks, {"object_instances": instances})],
        torch.device("cpu"),
        config,
        1,
    )

    assert result["best_threshold"] == 0.7
    assert result["quality_f1"] == 1.0
    assert result["best_threshold_object_f1"] == 1.0
    assert result["best_threshold_pixel_f1"] == 1.0


def test_object_threshold_counts_are_batched_without_changing_results() -> None:
    import numpy as np

    from mlsystem2.train import _trainer

    class RecordingExecutor:
        task_count = 0

        def map(self, function, tasks):
            materialized_tasks = list(tasks)
            self.task_count = len(materialized_tasks)
            return map(function, materialized_tasks)

    true_instances = np.zeros((2, 4, 4), dtype=np.int64)
    true_instances[0, 1:3, 1:3] = 1
    probabilities = np.zeros((2, 4, 4), dtype=np.float32)
    probabilities[0, 1:3, 1:3] = 0.8
    probabilities[1, 0, 0] = 0.8
    counts = {
        0.5: {"tp": 0, "fp": 0, "fn": 0},
        0.9: {"tp": 0, "fp": 0, "fn": 0},
    }
    executor = RecordingExecutor()

    _trainer._accumulate_object_threshold_counts(
        counts,
        true_instances,
        probabilities,
        executor,
    )

    assert executor.task_count == 4
    assert counts[0.5] == {"tp": 1, "fp": 1, "fn": 0}
    assert counts[0.9] == {"tp": 0, "fp": 0, "fn": 1}


def test_object_quality_requires_instance_masks() -> None:
    torch = pytest.importorskip("torch")

    class IdentityModel(torch.nn.Module):
        def forward(self, images):
            return images

    from mlsystem2.train import _trainer

    logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    masks = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    config = TrainConfig(
        quality_metric="objects",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=1,
    )

    with pytest.raises(TrainError, match="маски экземпляров"):
        _trainer._validate_epoch(
            torch,
            IdentityModel(),
            [(logits, masks)],
            torch.device("cpu"),
            config,
            1,
        )


def test_checkpoint_score_uses_best_threshold_pixel_f1() -> None:
    from mlsystem2.train import _trainer

    metrics = EpochMetrics(
        epoch=1,
        train_loss=1.0,
        val_loss=1.0,
        val_best_threshold=0.8,
        val_best_threshold_pixel_f1=0.7,
        val_best_threshold_precision=0.8,
        val_best_threshold_recall=0.62,
        epoch_time_sec=1.0,
    )

    assert _trainer._checkpoint_score(metrics) == 0.7


def test_object_quality_drives_checkpoint_and_early_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from mlsystem2.train import _trainer

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, images):
            return images * self.weight

    quality_scores = [0.9, 0.8, 0.7]
    pixel_scores = [0.1, 0.95, 0.99]

    monkeypatch.setattr(_trainer, "_train_epoch", lambda *args, **kwargs: {"loss": 1.0})

    def fake_validate(*args, **kwargs):
        epoch = int(args[-1])
        return {
            "loss": 1.0,
            "best_threshold": 0.7,
            "best_pixel_threshold": 0.5,
            "best_threshold_pixel_f1": pixel_scores[epoch - 1],
            "best_threshold_pixel_precision": pixel_scores[epoch - 1],
            "best_threshold_pixel_recall": pixel_scores[epoch - 1],
            "best_threshold_precision": quality_scores[epoch - 1],
            "best_threshold_recall": quality_scores[epoch - 1],
            "best_threshold_object_f1": quality_scores[epoch - 1],
            "best_threshold_object_precision": quality_scores[epoch - 1],
            "best_threshold_object_recall": quality_scores[epoch - 1],
            "quality_f1": quality_scores[epoch - 1],
            "quality_precision": quality_scores[epoch - 1],
            "quality_recall": quality_scores[epoch - 1],
        }

    saved: list[tuple[str, float]] = []
    monkeypatch.setattr(_trainer, "_validate_epoch", fake_validate)
    monkeypatch.setattr(
        _trainer,
        "_save_training_checkpoint",
        lambda request, path, metrics, label: saved.append((label, metrics.val_quality_f1)),
    )
    request = TrainRequest(
        model=ModelHandle(
            spec=ModelSpec(name="segformer_b0", input_channels=1, output_channels=1),
            model=TinyModel(),
        ),
        train_loader=[],
        val_loader=[],
        config=TrainConfig(
            quality_metric="objects",
            epochs=3,
            batch_size=1,
            device="cpu",
            learning_rate=0.001,
            weight_decay=0.0,
            loss="bce_dice",
            threshold=0.5,
            early_stopping_patience=1,
        ),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    result = train_model(request)

    assert result.epochs_total == 2
    assert saved == [("best", 0.9), ("final", 0.8)]


def test_object_training_checkpoint_contains_both_metric_families(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    instances = torch.zeros((2, 16, 16), dtype=torch.long)
    instances[:, 4:8, 4:8] = 1
    images, masks = _fake_loader(torch)[0]
    result = train_model(
        TrainRequest(
            model=ModelHandle(
                spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
                model=TinySegmentationModel(),
            ),
            train_loader=_fake_loader(torch),
            val_loader=[(images, masks, {"object_instances": instances})],
            config=TrainConfig(
                quality_metric="objects",
                epochs=1,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
            ),
            checkpoint_dir=str(tmp_path / "object-checkpoints"),
        )
    )

    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    metadata = checkpoint["metadata"]
    assert metadata["quality_metric"] == "objects"
    assert metadata["val_quality_f1"] == metadata["val_best_threshold_object_f1"]
    assert metadata["val_best_threshold_pixel_f1"] is not None
    assert metadata["val_best_threshold_pixel_precision"] is not None


def test_multiclass_rejects_object_quality_metric() -> None:
    with pytest.raises(ValueError, match="только для binary"):
        TrainConfig(
            task="multiclass",
            quality_metric="objects",
            epochs=1,
            batch_size=1,
            device="cpu",
            learning_rate=0.001,
            weight_decay=0.0,
            loss="cross_entropy",
            threshold=0.5,
            early_stopping_patience=1,
            class_slugs=["class_a"],
        )


def test_focal_tversky_loss_is_focal_plus_tversky() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.tensor([[[[-1.0, 0.5], [1.5, -0.25]]]], dtype=torch.float32)
    masks = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="focal_tversky",
        focal_alpha=0.6,
        pos_weight=1.7,
        tversky_alpha=0.4,
        tversky_beta=0.6,
        threshold=0.5,
        early_stopping_patience=1,
    )

    loss = _trainer._loss(torch, logits, masks, config)
    pos_weight = torch.tensor([config.pos_weight], dtype=logits.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(masks > 0.5, probs, 1.0 - probs)
    alpha_factor = torch.where(
        masks > 0.5,
        torch.as_tensor(config.focal_alpha, dtype=logits.dtype),
        torch.as_tensor(1.0 - config.focal_alpha, dtype=logits.dtype),
    )
    focal = (alpha_factor * torch.pow((1.0 - pt).clamp_min(0.0), 2.0) * bce).mean()
    true_positive = torch.sum(probs * masks)
    false_positive = torch.sum(probs * (1.0 - masks))
    false_negative = torch.sum((1.0 - probs) * masks)
    tversky = 1.0 - (true_positive + 1.0) / (
        true_positive
        + config.tversky_alpha * false_positive
        + config.tversky_beta * false_negative
        + 1.0
    )

    assert torch.allclose(loss, focal + tversky)
    assert not torch.allclose(loss, torch.pow(tversky, 2.0))


def test_hard_negative_weight_penalizes_hard_negative_false_positive_pixels() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.full((1, 1, 2, 2), 2.0, dtype=torch.float32)
    supervision = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    supervision[0, 0, 0, 0] = HARD_NEGATIVE_LABEL
    base_config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        focal_alpha=0.6,
        pos_weight=1.0,
        hard_negative_weight=1.0,
        tversky_alpha=0.4,
        tversky_beta=0.6,
        threshold=0.5,
        early_stopping_patience=1,
    )
    hard_config = base_config.model_copy(update={"hard_negative_weight": 3.0})
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        hard_config,
        torch.device("cpu"),
    )

    base_loss = _trainer._loss(torch, logits, masks, base_config, hard_negative_pixels)
    hard_loss = _trainer._loss(torch, logits, masks, hard_config, hard_negative_pixels)
    weights = _trainer._pixel_loss_weights(torch, logits, hard_negative_pixels, hard_config)

    assert hard_loss > base_loss
    assert torch.equal(masks, torch.zeros_like(masks))
    assert weights[0, 0, 0, 0].item() == pytest.approx(3.0)
    assert weights[0, 0, 0, 1].item() == pytest.approx(1.0)


def test_hard_negative_weight_penalizes_multiclass_hard_negative_foreground_pixels() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    logits[:, 1, :, :] = 2.0
    supervision = torch.zeros((1, 2, 2), dtype=torch.long)
    supervision[0, 0, 0] = HARD_NEGATIVE_LABEL
    base_config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="cross_entropy",
        focal_alpha=0.6,
        pos_weight=1.0,
        hard_negative_weight=1.0,
        tversky_alpha=0.4,
        tversky_beta=0.6,
        threshold=0.5,
        early_stopping_patience=1,
        class_slugs=["class_a"],
    )
    hard_config = base_config.model_copy(update={"hard_negative_weight": 3.0})
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        hard_config,
        torch.device("cpu"),
    )

    base_loss = _trainer._loss(torch, logits, masks, base_config, hard_negative_pixels)
    hard_loss = _trainer._loss(torch, logits, masks, hard_config, hard_negative_pixels)
    weights = _trainer._pixel_loss_weights(torch, logits, hard_negative_pixels, hard_config)

    assert hard_loss > base_loss
    assert torch.equal(masks, torch.zeros_like(masks))
    assert weights[0, 0, 0].item() == pytest.approx(3.0)
    assert weights[0, 0, 1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "cross_entropy_dice"])
def test_multiclass_class_hard_negative_penalizes_only_its_object_type(
    loss_name: str,
) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        hard_negative_weight=2.0,
        early_stopping_patience=1,
        class_slugs=["class_a", "class_b"],
    )
    masks = torch.zeros((1, 1, 1), dtype=torch.long)
    global_hard_negative = torch.zeros_like(masks, dtype=torch.bool)
    class_hard_negative = torch.tensor([[[[True]], [[False]]]])
    predicts_a = torch.tensor([[[[-4.0]], [[4.0]], [[-4.0]]]], requires_grad=True)
    predicts_b = torch.tensor([[[[-4.0]], [[-4.0]], [[4.0]]]], requires_grad=True)

    loss_a = _trainer._loss(
        torch,
        predicts_a,
        masks,
        config,
        global_hard_negative,
        class_hard_negative,
    )
    loss_b = _trainer._loss(
        torch,
        predicts_b,
        masks,
        config,
        global_hard_negative,
        class_hard_negative,
    )
    loss_a.backward()

    assert loss_a > loss_b
    assert predicts_a.grad is not None
    assert predicts_a.grad[0, 1, 0, 0].abs().item() > 0


def test_hard_negative_weight_ignores_tile_meta_without_supervision_pixels() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.full((1, 1, 2, 2), 2.0, dtype=torch.float32)
    masks = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        hard_negative_weight=3.0,
        threshold=0.5,
        early_stopping_patience=1,
    )

    loss_without_meta = _trainer._loss(torch, logits, masks, config, None)
    loss_with_old_meta = _trainer._loss(
        torch,
        logits,
        masks,
        config,
        {"tile_hard_negative": [True]},
    )

    assert torch.allclose(loss_without_meta, loss_with_old_meta)


def test_positive_pixels_do_not_receive_hard_negative_weight() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    supervision = torch.tensor(
        [[[[HARD_NEGATIVE_LABEL, 1.0], [0.0, 0.0]]]],
        dtype=torch.float32,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        hard_negative_weight=4.0,
        threshold=0.5,
        early_stopping_patience=1,
    )

    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )
    weights = _trainer._pixel_loss_weights(torch, logits, hard_negative_pixels, config)

    assert masks[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert masks[0, 0, 0, 1].item() == pytest.approx(1.0)
    assert weights[0, 0, 0, 0].item() == pytest.approx(4.0)
    assert weights[0, 0, 0, 1].item() == pytest.approx(1.0)


def test_hard_negative_weight_one_matches_base_loss() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.tensor([[[[1.5, 1.5], [0.0, 0.0]]]], dtype=torch.float32)
    supervision = torch.tensor(
        [[[[HARD_NEGATIVE_LABEL, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        hard_negative_weight=1.0,
        threshold=0.5,
        early_stopping_patience=1,
    )
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )

    base_loss = _trainer._loss(torch, logits, masks, config, None)
    hard_loss = _trainer._loss(torch, logits, masks, config, hard_negative_pixels)

    assert torch.allclose(base_loss, hard_loss)


@pytest.mark.parametrize("loss_name", ["bce_dice", "focal_dice", "focal_tversky"])
def test_binary_losses_support_pixel_hard_negative_weight(loss_name: str) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.randn((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    supervision = torch.tensor(
        [[[[HARD_NEGATIVE_LABEL, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        focal_alpha=0.6,
        hard_negative_weight=2.5,
        threshold=0.5,
        early_stopping_patience=1,
    )
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )

    loss = _trainer._loss(torch, logits, masks, config, hard_negative_pixels)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("loss_name", ["cross_entropy", "cross_entropy_dice"])
def test_multiclass_losses_support_pixel_hard_negative_weight(loss_name: str) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.randn((1, 3, 2, 2), dtype=torch.float32, requires_grad=True)
    supervision = torch.tensor([[[HARD_NEGATIVE_LABEL, 0], [1, 2]]], dtype=torch.long)
    config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        hard_negative_weight=2.5,
        threshold=0.5,
        early_stopping_patience=1,
        class_slugs=["class_a", "class_b"],
    )
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )

    loss = _trainer._loss(torch, logits, masks, config, hard_negative_pixels)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_train_model_skips_nonfinite_gradient_batch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model_impl = TinySegmentationModel()
    hook_calls = 0

    def first_gradient_is_nan(gradient):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            return torch.full_like(gradient, float("nan"))
        return gradient

    model_impl.conv.weight.register_hook(first_gradient_is_nan)
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=model_impl,
    )

    result = train_model(
        TrainRequest(
            model=model,
            train_loader=_fake_loader(torch),
            val_loader=_fake_loader(torch),
            config=TrainConfig(
                epochs=1,
                batch_size=2,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
            ),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
    )

    assert hook_calls >= 2
    assert result.epochs_total == 1
    assert len(result.history) == 1
    assert result.history[0].train_loss >= 0.0


def test_train_model_fails_after_second_nonfinite_gradient_batch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    model_impl = TinySegmentationModel()
    hook_calls = 0

    def first_two_gradients_are_nan(gradient):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls <= 2:
            return torch.full_like(gradient, float("nan"))
        return gradient

    model_impl.conv.weight.register_hook(first_two_gradients_are_nan)
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=model_impl,
    )

    with pytest.raises(TrainError, match="Слишком много non-finite gradients"):
        train_model(
            TrainRequest(
                model=model,
                train_loader=_fake_loader(torch),
                val_loader=_fake_loader(torch),
                config=TrainConfig(
                    epochs=1,
                    batch_size=2,
                    device="cpu",
                    learning_rate=0.001,
                    weight_decay=0.0,
                    loss="bce_dice",
                    threshold=0.5,
                    early_stopping_patience=1,
                ),
                checkpoint_dir=str(tmp_path / "checkpoints"),
            )
        )


@pytest.mark.parametrize("loss_name", ["bce_dice", "focal_dice", "focal_tversky"])
def test_binary_losses_penalize_false_positive_on_nodata_background(loss_name: str) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    supervision = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        focal_alpha=0.6,
        threshold=0.5,
        early_stopping_patience=1,
    )
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )
    low = torch.zeros((1, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    high = low.detach().clone().requires_grad_(True)
    low.data[0, 0, 0, 0] = -20.0
    high.data[0, 0, 0, 0] = 20.0

    low_loss = _trainer._loss(torch, low, masks, config, hard_negative_pixels)
    high_loss = _trainer._loss(torch, high, masks, config, hard_negative_pixels)

    assert high_loss > low_loss
    high_loss.backward()
    assert high.grad is not None
    assert abs(high.grad[0, 0, 0, 0].item()) > 0.0


@pytest.mark.parametrize("loss_name", ["cross_entropy", "cross_entropy_dice"])
def test_multiclass_losses_penalize_false_positive_on_nodata_background(
    loss_name: str,
) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    supervision = torch.tensor(
        [[[0, 0], [1, 2]]],
        dtype=torch.long,
    )
    config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        threshold=0.5,
        early_stopping_patience=1,
        class_slugs=["class_a", "class_b"],
    )
    masks, hard_negative_pixels = _trainer._prepare_supervision_masks(
        torch,
        supervision,
        config,
        torch.device("cpu"),
    )
    first = torch.zeros((1, 3, 2, 2), dtype=torch.float32, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    first.data[0, :, 0, 0] = torch.tensor([20.0, -20.0, -20.0])
    second.data[0, :, 0, 0] = torch.tensor([-20.0, 20.0, -20.0])

    first_loss = _trainer._loss(torch, first, masks, config, hard_negative_pixels)
    second_loss = _trainer._loss(torch, second, masks, config, hard_negative_pixels)

    assert second_loss > first_loss
    second_loss.backward()
    assert second.grad is not None
    assert torch.sum(torch.abs(second.grad[0, :, 0, 0])).item() > 0.0


def test_validation_loss_f1_and_threshold_ignore_context_frame() -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    class BorderArtifactModel(torch.nn.Module):
        def forward(self, images):
            logits = torch.full_like(images[:, :1], -20.0)
            logits[:, :, 0, :] = 20.0
            logits[:, :, -1, :] = 20.0
            logits[:, :, :, 0] = 20.0
            logits[:, :, :, -1] = 20.0
            logits[:, :, 2, 2] = 20.0
            return logits

    images = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    masks = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    masks[:, :, 2, 2] = 1.0

    def evaluate(context: int):
        return _trainer._validate_epoch(
            torch,
            BorderArtifactModel(),
            [(images, masks)],
            torch.device("cpu"),
            TrainConfig(
                epochs=1,
                batch_size=1,
                inference_context=context,
                device="cpu",
                learning_rate=0.001,
                weight_decay=0.0,
                loss="bce_dice",
                threshold=0.5,
                early_stopping_patience=1,
            ),
            1,
        )

    full = evaluate(0)
    central = evaluate(1)

    assert central["loss"] < full["loss"]
    assert central["best_threshold_pixel_f1"] == pytest.approx(1.0)
    assert central["best_threshold"] in _trainer.THRESHOLD_CANDIDATES
    assert full["best_threshold_pixel_f1"] < 0.2


@pytest.mark.parametrize("loss_name", ["bce_dice", "focal_dice", "focal_tversky"])
def test_binary_loss_has_zero_gradient_in_context_frame(loss_name: str) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.zeros((1, 1, 6, 6), dtype=torch.float32, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[:, :, 2:4, 2:4] = 1.0
    hard_negative = torch.zeros_like(logits, dtype=torch.bool)
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        inference_context=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        threshold=0.5,
        early_stopping_patience=1,
    )
    cropped_logits, cropped_masks, cropped_hard_negative = (
        _trainer._crop_supervision_tensors(logits, masks, hard_negative, 1)
    )

    _trainer._loss(
        torch,
        cropped_logits,
        cropped_masks,
        config,
        cropped_hard_negative,
    ).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :, 0, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, -1, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 1:-1, 1:-1]) > 0


@pytest.mark.parametrize("loss_name", ["cross_entropy", "cross_entropy_dice"])
def test_multiclass_loss_has_zero_gradient_in_context_frame(loss_name: str) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train import _trainer

    logits = torch.zeros((1, 3, 6, 6), dtype=torch.float32, requires_grad=True)
    masks = torch.zeros((1, 6, 6), dtype=torch.long)
    masks[:, 2:4, 2:4] = 1
    hard_negative = torch.zeros_like(masks, dtype=torch.bool)
    config = TrainConfig(
        task="multiclass",
        epochs=1,
        batch_size=1,
        inference_context=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss=loss_name,
        threshold=0.5,
        early_stopping_patience=1,
        class_slugs=["first", "second"],
    )
    cropped_logits, cropped_masks, cropped_hard_negative = (
        _trainer._crop_supervision_tensors(logits, masks, hard_negative, 1)
    )

    _trainer._loss(
        torch,
        cropped_logits,
        cropped_masks,
        config,
        cropped_hard_negative,
    ).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :, 0, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, -1, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, 0]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 1:-1, 1:-1]) > 0


def test_two_short_trainings_are_reproducible_with_same_seed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from mlsystem2.train_pipeline import _runner

    class TinySegmentationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, kernel_size=1)

        def forward(self, images):
            return self.conv(images)

    def run_once(name: str):
        _runner._seed_training(42)
        model = TinySegmentationModel()
        result = train_model(
            TrainRequest(
                model=ModelHandle(
                    spec=ModelSpec(
                        name="segformer_b2",
                        input_channels=4,
                        output_channels=1,
                    ),
                    model=model,
                ),
                train_loader=_fake_loader(torch),
                val_loader=_fake_loader(torch),
                config=TrainConfig(
                    epochs=2,
                    batch_size=2,
                    seed=42,
                    device="cpu",
                    learning_rate=0.001,
                    weight_decay=0.0,
                    loss="bce_dice",
                    threshold=0.5,
                    early_stopping_patience=2,
                ),
                checkpoint_dir=str(tmp_path / name),
                sample_size=16,
            )
        )
        return (
            {key: value.detach().clone() for key, value in model.state_dict().items()},
            [(item.train_loss, item.val_loss, item.val_quality_f1) for item in result.history],
        )

    first_state, first_history = run_once("first")
    second_state, second_history = run_once("second")

    assert first_history == pytest.approx(second_history)
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)


def test_training_pause_controller_preserves_process_and_optimizer_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from mlsystem2.train import _trainer

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones((1, 2))).sum()
    loss.backward()
    optimizer.step()
    state_before = {
        key: value.detach().clone()
        for state in optimizer.state.values()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    request_path = control_dir / _trainer.PAUSE_REQUEST_FILE
    marker_path = control_dir / _trainer.PAUSED_MARKER_FILE
    request_path.write_text("urgent-token\n", encoding="utf-8")
    controller = _trainer._TrainingPauseController(
        torch,
        model,
        optimizer,
        torch.device("cpu"),
        str(control_dir),
    )
    thread = threading.Thread(target=controller.pause_if_requested)
    thread.start()
    deadline = time.monotonic() + 3
    while not marker_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker_path.read_text(encoding="utf-8").strip() == "urgent-token"
    assert thread.is_alive()

    request_path.unlink()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert not marker_path.exists()
    state_after = {
        key: value.detach().clone()
        for state in optimizer.state.values()
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    assert state_before.keys() == state_after.keys()
    assert all(torch.equal(state_before[key], state_after[key]) for key in state_before)


def _fake_loader(torch, *, with_meta: bool = False):
    images = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    masks = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    masks[:, :, 4:8, 4:8] = 1.0
    if with_meta:
        return [
            (images, masks, {"augmented_tile_count": 2}),
            (images + 0.1, masks, {"augmented_tile_count": 1}),
        ]
    return [(images, masks), (images + 0.1, masks)]


def _fake_multiclass_loader(torch):
    images = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    masks = torch.zeros((2, 16, 16), dtype=torch.long)
    masks[:, 2:6, 2:6] = 1
    masks[:, 9:13, 9:13] = 2
    return [
        (images, masks, {"positive_tile_count": 2}),
        (images + 0.1, masks, {"positive_tile_count": 2}),
    ]
