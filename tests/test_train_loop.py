from __future__ import annotations

from pathlib import Path

import pytest

from mlsystem2.models.contracts import ModelHandle, ModelSpec
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainError, TrainRequest


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
    assert len(result.history) == 1
    assert Path(result.best_checkpoint_path).is_file()
    assert Path(result.final_checkpoint_path).is_file()
    assert result.history[0].train_loss >= 0.0
    assert result.history[0].train_optimizer_steps == 2
    assert result.history[0].train_skipped_optimizer_steps == 0
    assert result.history[0].train_loss_bce is not None
    assert result.history[0].train_loss_dice is not None
    assert result.history[0].val_loss >= 0.0


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
    assert result.history[0].val_positive_pixels > 0


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

    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["positive_pixels"] == 2
    assert result["pred_positive_pixels"] == 2
    assert result["f1"] == 0.5
    assert result["best_threshold"] == 0.3
    assert result["best_threshold_pixel_f1"] == 0.5
    assert 0.0 <= result["prob_p50"] <= 1.0


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

    assert result["positive_pixels"] == 0
    assert result["pred_positive_pixels"] == 4
    assert result["true_positive"] == 0
    assert result["false_negative"] == 0
    assert result["f1"] == 0.0
    assert result["best_threshold_pixel_f1"] == 0.0


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
    assert result.history[0].train_skipped_optimizer_steps == 1
    assert result.history[0].train_optimizer_steps == 1


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
