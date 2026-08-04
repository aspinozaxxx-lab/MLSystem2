from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qgis.PyQt.QtCore import QCoreApplication, QEventLoop, QTimer, Qt
from qgis.PyQt.QtWidgets import QMainWindow, QPushButton
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRuleBasedRenderer,
)
from qgis.gui import QgsMapCanvas, QgsMessageBar

from mlsystem_plugin.api_client import APIClient
from mlsystem_plugin.geometry_splitter import split_geometry
from mlsystem_plugin.plugin import MLSystemPlugin
from mlsystem_plugin.review_session import ReviewSession, ReviewSessionError
from mlsystem_plugin.settings import SERVER_URL, SERVER_USERNAME


class QGISPluginTests(unittest.TestCase):
    """Headless-proverki na realnom QGIS runtime."""

    # Inicializiruet edinstvennyi QgsApplication dlya suite.
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QgsApplication.instance() or QgsApplication([], False)
        cls.app.initQgis()

    # Osvobozhdaet QGIS providers posle suite.
    @classmethod
    def tearDownClass(cls) -> None:
        QgsProject.instance().clear()
        cls.app.exitQgis()

    # Sozdaet izolirovannyi proekt i katalog.
    def setUp(self) -> None:
        QgsProject.instance().clear()
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    # Zakryvaet sloi i udalyaaet dostupnye vremennye fayly.
    def tearDown(self) -> None:
        QgsProject.instance().clear()
        self.temp_dir.cleanup()

    # Proveriaet nemedlennyi vozvrat upravleniya event loop.
    def test_api_start_is_non_blocking(self) -> None:
        client = APIClient()
        client.configure("http://127.0.0.1:9", 1_000)
        started = time.perf_counter()

        client.create_job(
            "class-1",
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "EPSG:4326",
        )

        self.assertLess(time.perf_counter() - started, 0.1)
        client.abort_all()
        QCoreApplication.processEvents()

    # Проверяет вход по учётной записи и передачу серверной cookie.
    def test_api_login_and_fetches_class_list(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ClassListHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = APIClient()
        client.configure(f"http://127.0.0.1:{server.server_port}", 2_000)
        loop = QEventLoop()
        received: list[object] = []

        def handle_success(operation: str, payload: object) -> None:
            if operation == "login":
                client.get_classes()
                return
            received.append(payload)
            loop.quit()

        client.succeeded.connect(handle_success)
        client.failed.connect(lambda *args: loop.quit())

        client.login("mluser", "secret")
        QTimer.singleShot(3_000, loop.quit)
        loop.exec()
        server.shutdown()
        server.server_close()

        self.assertEqual(received[0]["classes"][0]["class_id"], "class-1")

    # Proveriaet sozdanie dock i oblast globalnyh hotkeys.
    def test_plugin_opens_dock_and_disables_shortcuts_without_session(self) -> None:
        iface = _FakeIface()
        plugin = MLSystemPlugin(iface)

        plugin.initGui()

        self.assertIsNotNone(plugin.dock)
        self.assertEqual(SERVER_URL, "https://grovika.ru")
        self.assertEqual(SERVER_USERNAME, "mlsystem")
        button_texts = {button.text() for button in plugin.dock.findChildren(QPushButton)}
        self.assertNotIn("Подключиться", button_texts)
        self.assertNotIn("Текущий охват", button_texts)
        self.assertNotIn("Из выделения", button_texts)
        self.assertNotIn("Открыть сохранённую сессию…", button_texts)
        self.assertNotIn("Сопоставить поля", button_texts)
        self.assertNotIn("Применить результаты в рабочие слои", button_texts)
        self.assertNotIn("1 — В разметку", button_texts)
        self.assertNotIn("2 — Hard negative", button_texts)
        self.assertNotIn("3 — Выкинуть", button_texts)
        self.assertIn("Использовать эту AOI", button_texts)
        self.assertIn("Разбить текущий", button_texts)
        self.assertIn("Разбить все крупные", button_texts)
        self.assertIn("Выгрузить на новый слой", button_texts)
        self.assertFalse(hasattr(plugin.dock, "check_connection"))
        self.assertFalse(hasattr(plugin.dock, "aoi_layer_combo"))
        self.assertFalse(hasattr(plugin.dock, "annotation_layer_combo"))
        self.assertFalse(hasattr(plugin.dock, "hard_layer_combo"))
        self.assertFalse(hasattr(plugin.dock, "filter_combo"))
        self.assertFalse(hasattr(plugin.dock, "server_url"))
        self.assertFalse(hasattr(plugin.dock, "username"))
        self.assertFalse(hasattr(plugin.dock, "password"))
        self.assertEqual(len(plugin._shortcuts), 4)
        self.assertTrue(all(not shortcut.isEnabled() for shortcut in plugin._shortcuts))
        plugin._set_shortcuts_enabled(True)
        iface.mapCanvas().setFocus()
        self.assertTrue(all(shortcut.isEnabled() for shortcut in plugin._shortcuts))
        plugin._set_shortcuts_enabled(False)
        plugin.unload()

    # Проверяет автоматический вход без отдельной кнопки подключения.
    def test_plugin_connects_automatically_with_configured_account(self) -> None:
        iface = _FakeIface()
        with (
            patch("mlsystem_plugin.dock_widget.load_connection_password", return_value="secret"),
            patch("mlsystem_plugin.dock_widget.APIClient.login") as login,
        ):
            plugin = MLSystemPlugin(iface)
            plugin.initGui()
            iface.mainWindow().show()
            QCoreApplication.processEvents()
            QCoreApplication.processEvents()

        login.assert_called_once_with("mlsystem", "secret")
        self.assertTrue(plugin.dock._connecting)
        plugin.unload()

    # AOI применяется отдельной кнопкой, а правая кнопка сбрасывает новый черновик.
    def test_aoi_requires_explicit_use_and_right_click_resets_draft(self) -> None:
        iface = _FakeIface()
        iface.mapCanvas().setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        plugin = MLSystemPlugin(iface)
        plugin.initGui()
        plugin.dock._draw_aoi()
        plugin.dock._aoi_tool._points = [
            QgsPointXY(0, 0),
            QgsPointXY(1_000, 0),
            QgsPointXY(1_000, 1_000),
            QgsPointXY(0, 1_000),
        ]

        self.assertIsNone(plugin.dock._aoi_geometry)
        plugin.dock._use_aoi()

        self.assertIsNotNone(plugin.dock._aoi_geometry)
        self.assertIn("AOI задана; площадь:", plugin.dock.aoi_label.text())
        self.assertIsNotNone(plugin.dock._aoi_tool.last_capture())
        self.assertGreater(len(plugin.dock._aoi_tool._points), 0)

        QCoreApplication.processEvents()
        plugin.dock._draw_aoi()
        plugin.dock._aoi_tool._points = [QgsPointXY(0, 0), QgsPointXY(1, 0)]
        mouse_buttons = getattr(Qt, "MouseButton", Qt)
        plugin.dock._aoi_tool.canvasReleaseEvent(
            SimpleNamespace(button=lambda: mouse_buttons.RightButton)
        )

        self.assertIsNone(plugin.dock._aoi_geometry)
        self.assertEqual(plugin.dock._aoi_tool._points, [])
        self.assertIs(iface.mapCanvas().mapTool(), plugin.dock._aoi_tool)
        self.assertIn("рисуйте заново", plugin.dock.aoi_label.text())
        plugin.unload()

    # Показывает число реально выбранных сервером снимков для контроля покрытия AOI.
    def test_job_status_shows_selected_image_count(self) -> None:
        iface = _FakeIface()
        plugin = MLSystemPlugin(iface)
        plugin.initGui()

        plugin.dock._api_succeeded(
            "job_status",
            {
                "status": "running",
                "current_stage": "inference",
                "progress": 50,
                "warnings": [],
                "coverage_percent": 100.0,
                "source_image_ids": ["scene-a", "scene-b"],
            },
        )

        self.assertIn("Выбрано снимков: 2", plugin.dock.warning_label.text())
        self.assertIn("Покрытие AOI: 100.00%", plugin.dock.warning_label.text())
        plugin.unload()

    # Проверяет единые пороги очереди и фактически отображаемого слоя.
    def test_review_thresholds_filter_queue_and_layer_renderer(self) -> None:
        session = self._session()
        first_id, second_id = session.feature_ids()
        confidence_index = session.layer.fields().indexOf("confidence")
        area_index = session.layer.fields().indexOf("area_m2")
        self.assertTrue(
            session.layer.dataProvider().changeAttributeValues(
                {second_id: {confidence_index: 0.9, area_index: 100.0}}
            )
        )

        session.set_thresholds(500.0, 0.7)

        self.assertEqual(session.feature_ids(), [first_id])
        self.assertEqual(session.counts()["visible"], 1)
        renderer = session.layer.renderer()
        self.assertIsInstance(renderer, QgsRuleBasedRenderer)
        expressions = [rule.filterExpression() for rule in renderer.rootRule().children()]
        self.assertTrue(any('"area_m2"' in expression for expression in expressions))
        self.assertTrue(any('"confidence"' in expression for expression in expressions))

        session.set_thresholds(0.0, 0.8)
        self.assertEqual(session.feature_ids(), [second_id])

    # Проверяет цветную подсветку текущего кандидата и её снятие фильтром.
    def test_plugin_highlights_current_candidate(self) -> None:
        iface = _FakeIface()
        plugin = MLSystemPlugin(iface)
        plugin.initGui()
        plugin.dock.min_area.setValue(0.0)
        session = self._session(single=True)

        plugin.dock._set_session(session)

        self.assertTrue(plugin.dock._candidate_highlight.isVisible())
        plugin.dock.min_area.setValue(2_000_000.0)
        self.assertFalse(plugin.dock._candidate_highlight.isVisible())
        plugin.unload()

    # Запуск нового распознавания убирает даже не привязанную после перезапуска старую сессию.
    def test_starting_job_removes_stale_candidate_layers_but_keeps_session_file(self) -> None:
        stale_session = self._session()
        stale_layer_id = stale_session.layer_id
        stale_path = stale_session.path
        iface = _FakeIface()
        plugin = MLSystemPlugin(iface)
        plugin.initGui()
        plugin.dock.class_combo.addItem("Реки", {"class_id": "rivers", "display_name": "Реки"})
        plugin.dock._set_aoi(
            QgsGeometry.fromWkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))"),
            QgsCoordinateReferenceSystem("EPSG:3857"),
        )
        requests: list[tuple[object, ...]] = []
        plugin.dock.api.create_job = lambda *args: requests.append(args)

        plugin.dock._start_job()

        self.assertIsNone(QgsProject.instance().mapLayer(stale_layer_id))
        self.assertTrue(stale_path.is_file())
        self.assertEqual(len(requests), 1)
        plugin.unload()

    # Проверяет площадь AOI без каких-либо рабочих целевых слоёв.
    def test_plugin_shows_aoi_area_without_target_layer_controls(self) -> None:
        iface = _FakeIface()
        plugin = MLSystemPlugin(iface)
        plugin.initGui()

        plugin.dock._set_aoi(
            QgsGeometry.fromWkt("POLYGON((0 0, 1000 0, 1000 1000, 0 1000, 0 0))"),
            QgsCoordinateReferenceSystem("EPSG:3857"),
        )

        self.assertFalse(hasattr(plugin.dock, "annotation_layer_combo"))
        self.assertFalse(hasattr(plugin.dock, "hard_layer_combo"))
        self.assertIn("площадь:", plugin.dock.aoi_label.text())
        self.assertIn("м²", plugin.dock.aoi_label.text())
        plugin.unload()

    # Proveriaet atomarnost, roditelya i balans ploshchadi split.
    def test_split_is_atomic_and_preserves_area_and_parent(self) -> None:
        session = self._session(single=True, large=True)
        parent = session.current_feature()
        parent_id = int(parent.id())
        parent_candidate_id = str(parent["candidate_id"])
        source_parts = split_geometry(
            parent.geometry(),
            session.layer.crs(),
            parent_candidate_id,
            4_000_000.0,
            1.0,
        )
        source_area = sum(part.area_m2 for part in source_parts)

        session.split_current(4_000_000.0)

        self.assertEqual(session.feature(parent_id)["review_status"], "split")
        children = [
            feature
            for feature in session.layer.getFeatures()
            if feature["parent_candidate_id"] == parent_candidate_id
        ]
        self.assertGreaterEqual(len(children), 2)
        self.assertEqual(len({feature["candidate_id"] for feature in children}), len(children))
        self.assertLessEqual(
            abs(source_area - sum(float(feature["area_m2"]) for feature in children)) / source_area,
            0.01,
        )

    # Выгружает только прошедшие фильтры части и не переносит служебные поля.
    def test_split_all_and_export_use_filtered_active_parts_only(self) -> None:
        session = self._session(large=True)
        first_id, second_id = session.feature_ids()
        confidence_index = session.layer.fields().indexOf("confidence")
        area_index = session.layer.fields().indexOf("area_m2")
        self.assertTrue(
            session.layer.dataProvider().changeAttributeValues(
                {
                    first_id: {area_index: 20_000_000.0},
                    second_id: {area_index: 20_000_000.0, confidence_index: 0.4},
                }
            )
        )
        session.set_thresholds(0.0, 0.7)

        split_count = session.split_large_candidates(4_000_000.0)
        output = session.export_filtered_layer("Реки MLSystem2")

        self.assertEqual(split_count, 1)
        self.assertGreater(output.featureCount(), 1)
        self.assertIsNotNone(QgsProject.instance().mapLayer(output.id()))
        self.assertEqual(output.fields().indexOf("review_status"), -1)
        self.assertEqual(output.fields().indexOf("target_layer_id"), -1)
        self.assertTrue(
            all(feature["parent_candidate_id"] == "candidate-1" for feature in output.getFeatures())
        )
        self.assertEqual(session.feature(second_id)["review_status"], "new")

    # Проверяет ошибку доступа после внешнего удаления слоя результатов.
    def test_removed_session_layer_fails_cleanly(self) -> None:
        session = self._session(single=True)
        feature_id = session.feature_ids()[0]
        QgsProject.instance().removeMapLayer(session.layer.id())
        with self.assertRaises(ReviewSessionError):
            session.feature(feature_id)

    # Sozdaet postoyannuyu testovuyu review session.
    def _session(self, *, single: bool = False, large: bool = False) -> ReviewSession:
        size = 0.04 if large else 0.01
        features = [self._feature("candidate-1", 30.0, 60.0, size)]
        if not single:
            features.append(self._feature("candidate-2", 30.02, 60.0, size))
        payload = {"type": "FeatureCollection", "features": features}
        return ReviewSession.from_geojson(
            payload,
            "11111111-1111-1111-1111-111111111111",
            Path(self.temp_dir.name),
        )

    # Formiruet odin GeoJSON-kandidat.
    @staticmethod
    def _feature(candidate_id: str, x: float, y: float, size: float) -> dict[str, object]:
        geometry = QgsGeometry.fromWkt(
            f"POLYGON(({x} {y}, {x + size} {y}, {x + size} {y - size}, {x} {y - size}, {x} {y}))"
        )
        return {
            "type": "Feature",
            "geometry": json.loads(geometry.asJson()),
            "properties": {
                "candidate_id": candidate_id,
                "job_id": "job-1",
                "class_id": "class-1",
                "class_name": "Опустынивание",
                "confidence": 0.75,
                "model_id": "model-1",
                "model_version": "run-1",
                "source_image_ids": ["scene-1"],
                "area_m2": 1_000_000.0,
            },
        }

class _FakeIface:
    """Minimalnyi QGIS iface dlya proverki zhiznennogo cikla UI."""

    # Sozdaet glavnoe okno, canvas i message bar.
    def __init__(self) -> None:
        self._window = QMainWindow()
        self._canvas = QgsMapCanvas(self._window)
        self._message_bar = QgsMessageBar(self._window)

    # Vozvrashchaet glavnoe okno QGIS.
    def mainWindow(self):  # noqa: N802
        return self._window

    # Vozvrashchaet map canvas.
    def mapCanvas(self):  # noqa: N802
        return self._canvas

    # Vozvrashchaet message bar.
    def messageBar(self):  # noqa: N802
        return self._message_bar

    # Dobavlyaet dock v testovoe okno.
    def addDockWidget(self, area, dock):  # noqa: N802
        self._window.addDockWidget(area, dock)

    # Ubiraet dock iz testovogo okna.
    def removeDockWidget(self, dock):  # noqa: N802
        self._window.removeDockWidget(dock)

    # Imitiruet registraciyu punkta menu.
    def addPluginToVectorMenu(self, menu, action):  # noqa: N802
        del menu, action

    # Imitiruet udalenie punkta menu.
    def removePluginVectorMenu(self, menu, action):  # noqa: N802
        del menu, action


class _ClassListHandler(BaseHTTPRequestHandler):
    """Minimalnyi lokalnyi server dlya proverki Qt network."""

    # Создаёт тестовую cookie после проверки обычной учётной записи.
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/api/v1/auth/login" or payload != {
            "username": "mluser",
            "password": "secret",
        }:
            self._json_response(401, {"detail": "Неверный логин или пароль"})
            return
        self._json_response(
            200,
            {"status": "ok"},
            {"Set-Cookie": "mlsystem2_session=test-session; Path=/; HttpOnly"},
        )

    # Vozvrashchaet testovyi spisok klassov.
    def do_GET(self) -> None:  # noqa: N802
        if "mlsystem2_session=test-session" not in self.headers.get("Cookie", ""):
            self._json_response(401, {"detail": "Требуется авторизация"})
            return
        self._json_response(200, {"classes": [{"class_id": "class-1"}]})

    # Отправляет компактный JSON-ответ локального сервера.
    def _json_response(
        self,
        status: int,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    # Podavlyaet standartnyi stderr-log testovogo servera.
    def log_message(self, format, *args) -> None:
        del format, args


if __name__ == "__main__":
    unittest.main(verbosity=2)
