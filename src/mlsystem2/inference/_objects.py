"""Общая сборка вероятностей и разделение объектов для обучения и инференса."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage

from .contracts import InferenceError, ObjectSceneResult, ObjectSeparationConfig


def separate_objects(foreground, boundary, valid_pixels, config: ObjectSeparationConfig):
    from skimage.segmentation import watershed

    foreground = np.asarray(foreground)
    boundary = np.asarray(boundary)
    valid = np.asarray(valid_pixels, dtype=bool)
    if foreground.ndim != 2 or foreground.shape != boundary.shape or foreground.shape != valid.shape:
        raise InferenceError("Карты области, границ и валидности должны иметь одинаковый размер")
    if not np.isfinite(foreground).all() or not np.isfinite(boundary).all():
        raise InferenceError("Карты вероятностей содержат нечисловые значения")
    mask = (foreground >= config.foreground_threshold) & valid
    components, count = ndimage.label(mask)
    markers, _ = ndimage.label(mask & (boundary < config.boundary_threshold))
    sizes = np.bincount(markers.ravel())
    markers[sizes[markers] < config.min_marker_pixels] = 0
    next_marker = int(markers.max()) + 1
    for label_id, slices in enumerate(ndimage.find_objects(components, count), start=1):
        if slices is None:
            continue
        component = components[slices] == label_id
        if np.any(markers[slices][component]):
            continue
        # Нулевой ободок нужен и для компонента, занимающего весь bbox.
        distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
        point = np.unravel_index(np.argmax(distance), distance.shape)
        markers[slices][point] = next_marker
        next_marker += 1
    labels = watershed(boundary, markers, mask=mask, connectivity=1, watershed_line=False)
    _, inverse = np.unique(np.concatenate(([0], labels.ravel())), return_inverse=True)
    return inverse[1:].reshape(labels.shape).astype(np.int32)


class SceneAccumulator:
    def __init__(self, request):
        self.request = request
        self._temporary = TemporaryDirectory(prefix="object-f1-", dir=request.work_dir)
        self._arrays = []
        self._finished = False
        self.shape = (request.height, request.width)
        self.probabilities = self._array("probabilities", np.float32, (2, *self.shape))
        self.weights = self._array("weights", np.float32, self.shape)
        self.valid = self._array("valid", np.bool_, self.shape)
        axis = np.arange(request.tile_size, dtype=np.float32) - request.tile_size / 2
        gaussian = np.exp(-(axis ** 2) / (2 * (request.tile_size * 0.25) ** 2))
        self._gaussian = np.outer(gaussian, gaussian)

    def _array(self, name, dtype, shape):
        result = np.memmap(Path(self._temporary.name) / name, dtype=dtype, mode="w+", shape=shape)
        self._arrays.append(result)
        return result

    def add_window(self, window):
        if self._finished:
            raise InferenceError("Накопитель уже завершён")
        size = self.request.tile_size
        probabilities = np.asarray(window.probabilities, dtype=np.float32)
        valid = np.asarray(window.valid_pixels, dtype=bool)
        if probabilities.shape != (2, size, size) or valid.shape != (size, size):
            raise InferenceError("Окно object f1 должно содержать два канала и valid mask полного размера")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise InferenceError("Вероятности object f1 должны быть конечными числами от 0 до 1")
        x0, y0 = max(0, window.x), max(0, window.y)
        x1, y1 = min(self.request.width, window.x + size), min(self.request.height, window.y + size)
        if x1 <= x0 or y1 <= y0:
            raise InferenceError("Окно не пересекает снимок")
        source = np.s_[y0-window.y:y1-window.y, x0-window.x:x1-window.x]
        target = np.s_[y0:y1, x0:x1]
        weight = self._gaussian[source] * valid[source]
        self.probabilities[(slice(None), *target)] += probabilities[(slice(None), *source)] * weight
        self.weights[target] += weight
        self.valid[target] |= valid[source]

    def finish(self):
        if self._finished:
            raise InferenceError("Накопитель уже завершён")
        self._finished = True
        foreground = self._array("foreground", np.bool_, self.shape)
        components = self._array("components", np.int32, self.shape)
        instances = self._array("instances", np.int32, self.shape)
        for y in range(0, self.request.height, 256):
            strip = np.s_[y:y+256, :]
            weights = self.weights[strip]
            np.divide(self.probabilities[:, y:y+256], np.maximum(weights, 1e-12),
                      out=self.probabilities[:, y:y+256])
            foreground[strip] = (self.probabilities[0][strip] >= self.request.separation.foreground_threshold) & self.valid[strip]
        count = ndimage.label(foreground, output=components)
        offset = 0
        for component_id, slices in enumerate(ndimage.find_objects(components, count), start=1):
            if slices is None:
                continue
            mask = components[slices] == component_id
            labels = separate_objects(self.probabilities[0][slices], self.probabilities[1][slices], mask, self.request.separation)
            positive = labels > 0
            instances[slices][positive] = labels[positive] + offset
            offset += int(labels.max())
        return ObjectSceneResult(probabilities=self.probabilities, valid_pixels=self.valid,
                                 instances=instances, object_count=offset)

    def close(self):
        for array in self._arrays:
            array.flush()
            array._mmap.close()
        self._arrays.clear()
        self._temporary.cleanup()
