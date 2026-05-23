from __future__ import annotations

from pathlib import Path

import pytest

from mlsystem2.models.contracts import ModelHandle, ModelSpec
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest


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
    assert result.history[0].val_loss >= 0.0


def _fake_loader(torch):
    images = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    masks = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    masks[:, :, 4:8, 4:8] = 1.0
    return [(images, masks), (images + 0.1, masks)]
