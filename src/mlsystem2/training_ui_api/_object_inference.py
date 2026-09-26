"""Чтение растровых окон object f1 для PyTorch и адаптера Geoalert."""

import time

import numpy as np
from rasterio.windows import Window

from mlsystem2.inference.api import create_object_scene, object_window_origins
from mlsystem2.inference.contracts import ObjectSceneRequest, ObjectSeparationConfig, ObjectWindowPrediction


def predict_instances(dataset, predict, *, input_indexes, tile_size, batch_size=1, threshold=0.5, metrics=None):
    performance = metrics if metrics is not None else {}
    accumulator = create_object_scene(ObjectSceneRequest(width=dataset.width, height=dataset.height, tile_size=tile_size,
        separation=ObjectSeparationConfig(foreground_threshold=threshold)))
    windows = [(x, y) for y in object_window_origins(dataset.height, tile_size)
               for x in object_window_origins(dataset.width, tile_size)]
    try:
        for start in range(0, len(windows), batch_size):
            pending = windows[start:start+batch_size]
            images, validity = [], []
            started = time.perf_counter()
            for x, y in pending:
                window = Window(x, y, tile_size, tile_size)
                boundless = x + tile_size > dataset.width or y + tile_size > dataset.height
                images.append(dataset.read(indexes=input_indexes, window=window, boundless=boundless,
                    fill_value=0, out_dtype="float32", out_shape=(len(input_indexes), tile_size, tile_size)))
                validity.append(dataset.dataset_mask(window=window, boundless=boundless, out_shape=(tile_size, tile_size)) > 0)
            performance["reading_sec"] = float(performance.get("reading_sec", 0)) + time.perf_counter() - started
            started = time.perf_counter()
            probabilities = np.asarray(predict(np.stack(images)), dtype=np.float32)
            if probabilities.shape != (len(pending), 2, tile_size, tile_size):
                raise RuntimeError("object f1 ожидает две карты вероятностей на каждое окно")
            performance["prediction_sec"] = float(performance.get("prediction_sec", 0)) + time.perf_counter() - started
            for (x, y), prediction, valid in zip(pending, probabilities, validity, strict=True):
                accumulator.add_window(ObjectWindowPrediction(x=x, y=y, probabilities=prediction, valid_pixels=valid))
            performance["tile_count"] = int(performance.get("tile_count", 0)) + len(pending)
        started = time.perf_counter()
        result = accumulator.finish()
        performance["object_separation_sec"] = time.perf_counter() - started
        performance["object_count"] = result.object_count
        return accumulator, result
    except BaseException:
        accumulator.close()
        raise


def torch_object_predictor(torch, model, device, input_channels):
    def predict(images):
        if images.shape[1] == 3 and input_channels == 4:
            images = np.concatenate((images, np.zeros_like(images[:, :1])), axis=1)
        with torch.no_grad():
            logits = model(torch.as_tensor(images, dtype=torch.float32, device=device))
            if logits.ndim != 4 or logits.shape[1] != 3:
                raise RuntimeError("Checkpoint object f1 должен возвращать три logits")
            return torch.cat((logits[:, :2].softmax(1)[:, 1:2], logits[:, 2:3].sigmoid()), 1).cpu().numpy()
    return predict
