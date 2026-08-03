"""Instrument risovaniya AOI na karte QGIS."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand


# Vozvrashchaet sovmestimuyu konstantu knopki myshi Qt5/Qt6.
def _mouse_button(name: str):
    enum = getattr(Qt, "MouseButton", Qt)
    return getattr(enum, name)


# Vozvrashchaet sovmestimuyu konstantu klavishi Qt5/Qt6.
def _key(name: str):
    enum = getattr(Qt, "Key", Qt)
    return getattr(enum, name)


class AOIPolygonMapTool(QgsMapTool):
    """Sobiraet vershiny poligona levym klikom i zavershaet pravym."""

    captured = pyqtSignal(object, object)
    cancelled = pyqtSignal()

    # Inicializiruet map tool i ego rubber band.
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

    # Obrabatyvaet dobavlenie vershiny i zavershenie kontura.
    def canvasReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == _mouse_button("LeftButton"):
            point = self.toMapCoordinates(event.pos())
            self._points.append(QgsPointXY(point))
            self._redraw()
            return
        if event.button() == _mouse_button("RightButton"):
            self._finish()

    # Otmenyaet risovanie po Escape.
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == _key("Key_Escape"):
            self.reset()
            self.cancelled.emit()

    # Ochishchaet nezavershennyi kontur pri smene instrumenta.
    def deactivate(self) -> None:
        self.reset()
        super().deactivate()

    def reset(self) -> None:
        """Ochistit nezavershennyi kontur."""

        self._points.clear()
        self._rubber_band.reset(Qgis.GeometryType.Polygon)

    def last_capture(
        self,
    ) -> tuple[QgsGeometry, QgsCoordinateReferenceSystem] | None:
        """Вернуть копию последнего завершённого контура."""

        if self._last_geometry is None or self._last_crs is None:
            return None
        return QgsGeometry(self._last_geometry), QgsCoordinateReferenceSystem(self._last_crs)

    # Pererisovyvaet tekushchii kontur AOI.
    def _redraw(self) -> None:
        self._rubber_band.reset(Qgis.GeometryType.Polygon)
        for point in self._points:
            self._rubber_band.addPoint(point, False)
        if self._points:
            self._rubber_band.addPoint(self._points[0], True)

    # Zavershaet validnyi kontur i peredaet ego paneli.
    def _finish(self) -> None:
        if len(self._points) < 3:
            return
        ring = [*self._points, self._points[0]]
        geometry = QgsGeometry.fromPolygonXY([ring])
        crs = self._canvas.mapSettings().destinationCrs()
        self._last_geometry = QgsGeometry(geometry)
        self._last_crs = QgsCoordinateReferenceSystem(crs)
        self.reset()
        self.captured.emit(QgsGeometry(geometry), QgsCoordinateReferenceSystem(crs))


__all__ = ["AOIPolygonMapTool"]
