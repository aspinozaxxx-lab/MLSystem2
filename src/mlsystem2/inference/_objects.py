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
    if config.profile == "object_f1_v2":
        return _separate_objects_v2(foreground, boundary, valid, config)
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


def _separate_objects_v2(foreground, boundary, valid, config):
    """Восстановить полные объекты из устойчивых маркеров без возврата мелкого шума."""
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    mask = (foreground >= config.foreground_threshold) & valid
    components, count = ndimage.label(mask)
    sizes = np.bincount(components.ravel())
    instances = np.zeros(mask.shape, dtype=np.int32)
    offset = 0
    for component_id, slices in enumerate(ndimage.find_objects(components, count), start=1):
        if slices is None or sizes[component_id] < config.min_object_pixels:
            continue
        component = components[slices] == component_id
        values = boundary[slices]
        interior = foreground[slices] - values
        # Низкая вероятность границы ещё не означает внутренность участка:
        # фон или размытый контур не должны порождать отдельный экземпляр.
        support = component & (interior > values) & (interior > 1 - foreground[slices])
        if not support.any():
            continue
        barriers = (values >= config.boundary_threshold) & component
        radius = config.boundary_closing_radius
        if radius:
            # Продление ободка не даёт замыканию стереть шов у края bbox/снимка.
            padded = np.pad(barriers, radius, mode="edge")
            closed = ndimage.binary_closing(padded, structure=np.ones((2 * radius + 1,) * 2, bool))
            barriers |= closed[radius:-radius, radius:-radius] & component
        markers, _ = ndimage.label(support & ~barriers)
        marker_sizes = np.bincount(markers.ravel())
        marker_sizes[0] = 0
        markers[marker_sizes[markers] < config.min_marker_pixels] = 0
        marker_count = int((marker_sizes >= config.min_marker_pixels).sum())
        if not marker_count:
            distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
            # Резервный центр также требует подтверждённой внутренности:
            # расстояние измеряется по полной области, а не по узким семенам.
            peaks = h_maxima(distance, config.fallback_peak_prominence) & support
            markers, marker_count = ndimage.label(peaks)
            if not np.any(markers):
                markers[np.unravel_index(np.argmax(np.where(support, distance, -1)), distance.shape)] = 1
                marker_count = 1
        if marker_count == 1:
            # Единственный маркер заполняет всю связную область при любом рельефе.
            instances[slices][component] = offset + 1
            offset += 1
            continue
        # Равномерное расширение внутри полной области отдаёт контур соседним
        # сердцевинам по геодезическому расстоянию. Рельеф границы способен
        # присоединить длинную петлю к одному участку вместо ближайших соседей.
        elevation = np.zeros(component.shape, dtype=np.float32)
        labels = watershed(elevation, markers, mask=component, connectivity=1, watershed_line=False)
        label_sizes = np.bincount(labels.ravel())
        # Пропуски ID после фильтра маркеров не являются маленькими объектами:
        # раньше они запускали повторный watershed, хотя ни один маркер не менялся.
        too_small = (label_sizes > 0) & (label_sizes < config.min_object_pixels)
        too_small[0] = False
        if np.all(too_small[np.unique(labels[component])]):
            too_small[1 + np.argmax(label_sizes[1:])] = False
        if too_small.any():
            # Удалённые маленькие центры не оставляют искусственных дыр в участке.
            markers[too_small[labels]] = 0
            labels = watershed(elevation, markers, mask=component, connectivity=1, watershed_line=False)
        positive = labels > 0
        if positive.any():
            present = np.flatnonzero(np.bincount(labels.ravel())[1:]) + 1
            compact = np.zeros(int(present[-1]) + 1, dtype=np.int32)
            compact[present] = np.arange(1, len(present) + 1, dtype=np.int32)
            instances[slices][positive] = compact[labels[positive]] + offset
            offset += len(present)
    return instances


class SceneAccumulator:
    def __init__(self, request):
        self.request = request
        self._temporary = TemporaryDirectory(prefix="object-f1-", dir=request.work_dir)
        self._arrays = []
        self._finished = False
        self._closed = False
        self._component_threshold = None
        self._component_slices = None
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
        if self._finished or self._closed:
            raise InferenceError("Накопитель уже завершён")
        size = self.request.tile_size
        probabilities = np.asarray(window.probabilities, dtype=np.float32)
        valid = np.asarray(window.valid_pixels, dtype=bool)
        if probabilities.shape != (2, size, size) or valid.shape != (size, size):
            raise InferenceError("Окно object f1 должно содержать два канала и valid mask полного размера")
        if not np.isfinite(probabilities).all():
            raise InferenceError("Вероятности object f1 должны быть конечными числами от 0 до 1")
        # Сумма двух softmax-классов float32 может дать 1 + epsilon.
        # Допуск только для v2; реальные ошибки входа нельзя скрывать обрезкой.
        tolerance = 1e-6 if self.request.separation.profile == "object_f1_v2" else 0.0
        minimum, maximum = float(probabilities.min()), float(probabilities.max())
        if minimum < -tolerance or maximum > 1 + tolerance:
            raise InferenceError(
                "Вероятности object f1 должны быть конечными числами от 0 до 1: "
                f"минимум={minimum:.9g}, максимум={maximum:.9g}"
            )
        if tolerance and (minimum < 0 or maximum > 1):
            probabilities = np.clip(probabilities, 0, 1)
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
        if self._finished or self._closed:
            raise InferenceError("Накопитель уже завершён")
        self._finished = True
        self._foreground = self._array("foreground", np.bool_, self.shape)
        self._components = self._array("components", np.int32, self.shape)
        self._instances = self._array("instances", np.int32, self.shape)
        for y in range(0, self.request.height, 256):
            strip = np.s_[y:y+256, :]
            weights = self.weights[strip]
            np.divide(self.probabilities[:, y:y+256], np.maximum(weights, 1e-12),
                      out=self.probabilities[:, y:y+256])
            if self.request.separation.profile == "object_f1_v2":
                # Взвешенное среднее также накапливает погрешность float32.
                np.clip(self.probabilities[:, y:y+256], 0, 1,
                        out=self.probabilities[:, y:y+256])
        return self._segment(self.request.separation)

    def resegment(self, config: ObjectSeparationConfig):
        """Повторить разделение по готовым вероятностям, переиспользуя дисковые карты."""
        if not self._finished or self._closed:
            raise InferenceError("Повторное разделение доступно после finish и до close")
        return self._segment(ObjectSeparationConfig.model_validate(config))

    def _segment(self, config):
        reuse_components = (config.profile == "object_f1_v2"
                            and self._component_threshold == config.foreground_threshold)
        for y in range(0, self.request.height, 256):
            strip = np.s_[y:y+256, :]
            if not reuse_components:
                self._foreground[strip] = (self.probabilities[0][strip] >= config.foreground_threshold) & self.valid[strip]
            self._instances[strip] = 0
        if not reuse_components:
            count = ndimage.label(self._foreground, output=self._components)
            self._component_slices = ndimage.find_objects(self._components, count)
            self._component_threshold = config.foreground_threshold if config.profile == "object_f1_v2" else None
        offset = 0
        for component_id, slices in enumerate(self._component_slices, start=1):
            if slices is None:
                continue
            mask = self._components[slices] == component_id
            labels = separate_objects(self.probabilities[0][slices], self.probabilities[1][slices], mask, config)
            positive = labels > 0
            self._instances[slices][positive] = labels[positive] + offset
            offset += int(labels.max())
        return ObjectSceneResult(probabilities=self.probabilities, valid_pixels=self.valid,
                                 instances=self._instances, object_count=offset)

    def close(self):
        self._closed = True
        for array in self._arrays:
            array.flush()
            array._mmap.close()
        self._arrays.clear()
        self._temporary.cleanup()
