"""Инструмент рисования AOI на карте QGIS."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand


def _mouse_button(name: str):
    """Вернуть совместимую константу кнопки мыши Qt5/Qt6."""

    enum = getattr(Qt, "MouseButton", Qt)
    return getattr(enum, name)


def _key(name: str):
    """Вернуть совместимую константу клавиши Qt5/Qt6."""

    enum = getattr(Qt, "Key", Qt)
    return getattr(enum, name)


class AOIPolygonMapTool(QgsMapTool):
    """Рисует полигон слева, подтверждает кнопкой панели и сбрасывает справа."""

    captured = pyqtSignal(object, object)
    cancelled = pyqtSignal()

    def __init__(self, canvas) -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._points: list[QgsPointXY] = []
        self._last_geometry: QgsGeometry | None = None
        self._last_crs: QgsCoordinateReferenceSystem | None = None
        self._rubber_band = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self._rubber_band.setColor(QColor(255, 170, 0, 180))
        self._rubber_band.setFillColor(QColor(255, 170, 0, 45))
        self._rubber_band.setWidth(2)

    def canvasReleaseEvent(self, event) -> None:  # noqa: N802
        """Добавить вершину слева или очистить черновик справа."""

        if event.button() == _mouse_button("LeftButton"):
            point = self.toMapCoordinates(event.pos())
            self._points.append(QgsPointXY(point))
            self._redraw()
            return
        if event.button() == _mouse_button("RightButton"):
            self.cancel_current()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Сбросить текущий контур по Escape."""

        if event.key() == _key("Key_Escape"):
            self.cancel_current()

    def deactivate(self) -> None:
        """Оставить подтверждённую AOI видимой после смены инструмента."""

        super().deactivate()

    def reset(self) -> None:
        """Очистить контур перед новым рисованием без сигнала отмены."""

        self._points.clear()
        self._last_geometry = None
        self._last_crs = None
        self._rubber_band.reset(Qgis.GeometryType.Polygon)

    def cancel_current(self) -> None:
        """Сбросить AOI и продолжить рисование с первой вершины."""

        self.reset()
        self.cancelled.emit()

    def capture_current(self) -> bool:
        """Подтвердить нарисованный контур по явной команде панели."""

        if len(self._points) < 3:
            return False
        ring = [*self._points, self._points[0]]
        geometry = QgsGeometry.fromPolygonXY([ring])
        crs = self._canvas.mapSettings().destinationCrs()
        self._last_geometry = QgsGeometry(geometry)
        self._last_crs = QgsCoordinateReferenceSystem(crs)
        self._redraw()
        self.captured.emit(QgsGeometry(geometry), QgsCoordinateReferenceSystem(crs))
        return True

    def last_capture(
        self,
    ) -> tuple[QgsGeometry, QgsCoordinateReferenceSystem] | None:
        """Вернуть копию последнего подтверждённого контура."""

        if self._last_geometry is None or self._last_crs is None:
            return None
        return QgsGeometry(self._last_geometry), QgsCoordinateReferenceSystem(self._last_crs)

    def _redraw(self) -> None:
        """Перерисовать текущий контур AOI."""

        self._rubber_band.reset(Qgis.GeometryType.Polygon)
        for point in self._points:
            self._rubber_band.addPoint(point, False)
        if self._points:
            self._rubber_band.addPoint(self._points[0], True)


__all__ = ["AOIPolygonMapTool"]
