from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import time
import unittest
from pathlib import Path

from qgis.PyQt.QtCore import QCoreApplication, QEventLoop, QTimer
from qgis.PyQt.QtWidgets import QMainWindow
from qgis.core import (
    QgsApplication,
    QgsField,
    QgsGeometry,
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsMapCanvas, QgsMessageBar

from mlsystem_plugin.api_client import APIClient
from mlsystem_plugin.geometry_splitter import split_geometry
from mlsystem_plugin.layer_utils import LayerOperationError, apply_reviewed_candidates
from mlsystem_plugin.plugin import MLSystemPlugin
from mlsystem_plugin.review_session import ReviewSession, ReviewSessionError

try:
    from qgis.PyQt.QtCore import QMetaType

    STRING_TYPE = QMetaType.Type.QString
except (ImportError, AttributeError):
    from qgis.PyQt.QtCore import QVariant

    STRING_TYPE = QVariant.String


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
        client.configure("http://127.0.0.1:9", "secret", 1_000)
        started = time.perf_counter()

        client.create_job(
            "class-1",
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "EPSG:4326",
        )

        self.assertLess(time.perf_counter() - started, 0.1)
        client.abort_all()
        QCoreApplication.processEvents()

    # Proveriaet realnyi asinkhronnyi GET spiska klassov.
    def test_api_fetches_class_list(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ClassListHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = APIClient()
        client.configure(f"http://127.0.0.1:{server.server_port}", "secret", 2_000)
        loop = QEventLoop()
        received: list[object] = []
        client.succeeded.connect(lambda operation, payload: (received.append(payload), loop.quit()))
        client.failed.connect(lambda *args: loop.quit())

        client.get_classes()
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
        self.assertEqual(len(plugin._shortcuts), 9)
        self.assertTrue(all(not shortcut.isEnabled() for shortcut in plugin._shortcuts))
        plugin._set_shortcuts_enabled(True)
        iface.mapCanvas().setFocus()
        self.assertTrue(all(shortcut.isEnabled() for shortcut in plugin._shortcuts))
        plugin._set_shortcuts_enabled(False)
        plugin.unload()

    # Proveriaet tri kategorii, undo, redo i povtornoe otkrytie.
    def test_review_actions_undo_redo_and_persistence(self) -> None:
        session = self._session()
        first_id = session.current_feature().id()

        session.classify("annotation")
        self.assertEqual(session.feature(first_id)["review_status"], "annotation")
        session.undo_stack.undo()
        self.assertEqual(session.feature(first_id)["review_status"], "new")
        session.undo_stack.redo()
        self.assertEqual(session.feature(first_id)["review_status"], "annotation")
        session.set_filter("all")
        session.select_feature(first_id)
        session.classify("hard_negative")
        session.undo_stack.undo()
        self.assertEqual(session.feature(first_id)["review_status"], "annotation")
        session.classify("discarded")
        session.undo_stack.undo()
        self.assertEqual(session.feature(first_id)["review_status"], "annotation")

        reopened_path = session.path
        session.close(remove_layer=True)
        reopened = ReviewSession.open_existing(reopened_path)
        self.assertEqual(reopened.feature(first_id)["review_status"], "annotation")

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

        session.split_current(4_000_000.0, 1.0)

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
        session.undo_stack.undo()
        self.assertEqual(session.feature(parent_id)["review_status"], "new")
        self.assertFalse(
            any(
                feature["parent_candidate_id"] == parent_candidate_id
                for feature in session.layer.getFeatures()
            )
        )
        session.undo_stack.redo()
        self.assertEqual(session.feature(parent_id)["review_status"], "split")

    # Proveriaet yavnye sloi, transformaciyu CRS i idempotentnost.
    def test_apply_uses_explicit_layers_transforms_crs_and_is_idempotent(self) -> None:
        session = self._session()
        session.set_filter("all")
        first_id, second_id = session.feature_ids()
        session.select_feature(first_id)
        session.classify("annotation")
        session.select_feature(second_id)
        session.classify("hard_negative")
        annotation = self._target("Разметка", "EPSG:3857")
        hard = self._target("Hard negative", "EPSG:4326")
        mapping = {
            "class_id": "",
            "source": "",
            "confidence": "",
            "candidate_id": "candidate_id",
            "model_version": "",
        }

        first = apply_reviewed_candidates(session.layer, annotation, hard, mapping, mapping)
        second = apply_reviewed_candidates(session.layer, annotation, hard, mapping, mapping)

        self.assertEqual(first.added, 2)
        self.assertEqual(second.added, 0)
        self.assertEqual(second.existing, 2)
        self.assertEqual(annotation.featureCount(), 1)
        self.assertEqual(hard.featureCount(), 1)
        self.assertGreater(annotation.extent().xMaximum(), 100_000)

    # Proveriaet rollback validacii i udalenie sloya sessii.
    def test_apply_validation_leaves_targets_unchanged_and_removed_session_fails(self) -> None:
        session = self._session(single=True)
        session.classify("annotation")
        annotation = self._target("Разметка", "EPSG:4326")
        hard = self._target("Hard negative", "EPSG:4326")
        with self.assertRaises(LayerOperationError):
            apply_reviewed_candidates(session.layer, annotation, hard, {}, {})
        self.assertEqual(annotation.featureCount(), 0)
        self.assertEqual(hard.featureCount(), 0)

        session.set_filter("all")
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
                "confidence": 0.75,
                "model_id": "model-1",
                "model_version": "run-1",
                "source_image_ids": ["scene-1"],
                "area_m2": 1_000_000.0,
            },
        }

    # Sozdaet yavno ukazannyi redaktiruemyi celevoi sloi.
    @staticmethod
    def _target(name: str, crs: str) -> QgsVectorLayer:
        layer = QgsVectorLayer(f"MultiPolygon?crs={crs}", name, "memory")
        layer.dataProvider().addAttributes([QgsField("candidate_id", STRING_TYPE)])
        layer.updateFields()
        QgsProject.instance().addMapLayer(layer)
        return layer


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

    # Vozvrashchaet testovyi spisok klassov.
    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"classes": [{"class_id": "class-1"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Podavlyaet standartnyi stderr-log testovogo servera.
    def log_message(self, format, *args) -> None:
        del format, args


if __name__ == "__main__":
    unittest.main(verbosity=2)
