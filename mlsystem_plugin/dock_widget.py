"""Kompaktnaya dock-panel proverki psevdorazmetki."""

from __future__ import annotations

import json
from dataclasses import replace

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsCsException,
    QgsDistanceArea,
    QgsGeometry,
    QgsMessageLog,
    QgsProject,
    QgsVectorLayer,
)
from qgis.gui import QgsRubberBand

from .aoi_map_tool import AOIPolygonMapTool
from .api_client import APIClient
from .contracts import PluginContractError, validate_feature_collection
from .layer_utils import (
    LayerOperationError,
    apply_reviewed_candidates,
    polygon_layers,
)
from .review_session import ReviewSession, ReviewSessionError, is_review_session_layer
from .settings import (
    SERVER_URL,
    SERVER_USERNAME,
    load_connection_password,
    load_field_mapping,
    load_settings,
    save_field_mapping,
    save_settings,
    session_directory,
)


# Vozvrashchaet sovmestimyi kod prinyatiya dialoga Qt5/Qt6.
def _accepted_dialog_code():
    enum = getattr(QDialog, "DialogCode", QDialog)
    return enum.Accepted


# Perevodit mashinnyi status v korotkii russkii tekst.
def _status_text(value: str) -> str:
    return {
        "queued": "в очереди",
        "running": "выполняется",
        "succeeded": "успешно",
        "failed": "ошибка",
        "cancelled": "отменено",
    }.get(value, value)


# Perevodit etap runner v russkii tekst.
def _stage_text(value: str) -> str:
    return {
        "queued": "ожидание",
        "running": "выполнение",
        "selecting_images": "подбор снимков",
        "loading_model": "загрузка модели",
        "inference": "инференс",
        "vectorization": "векторизация",
        "succeeded": "готово",
        "failed": "ошибка",
        "cancelled": "отменено",
    }.get(value, value)


def _format_area(area_m2: float) -> str:
    """Показать площадь в удобной для масштаба единице."""

    if area_m2 >= 1_000_000.0:
        value = f"{area_m2 / 1_000_000.0:,.2f}".replace(",", " ").replace(".", ",")
        return f"{value} км²"
    value = f"{area_m2:,.0f}".replace(",", " ")
    return f"{value} м²"


def _is_hard_negative_layer(layer: QgsVectorLayer) -> bool:
    name = "".join(character for character in layer.name().casefold() if character.isalnum())
    return "hardnegative" in name or "негатив" in name


class FieldMappingDialog(QDialog):
    """Nastroika sopostavleniya bez izmeneniya skhemy sloya."""

    _LABELS = {
        "class_id": "Класс",
        "source": "Источник",
        "confidence": "Confidence",
        "candidate_id": "Candidate ID",
        "model_version": "Версия модели",
    }

    # Stroit dve nezavisimye formy po realnym polyam sloev.
    def __init__(
        self,
        annotation_layer: QgsVectorLayer,
        hard_negative_layer: QgsVectorLayer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Сопоставление полей")
        layout = QVBoxLayout(self)
        self._combos: dict[tuple[str, str], QComboBox] = {}
        for role, title, layer in (
            ("annotation", "Слой разметки", annotation_layer),
            ("hard_negative", "Слой hard_negative", hard_negative_layer),
        ):
            group = QGroupBox(title)
            form = QFormLayout(group)
            current = load_field_mapping(role, layer)
            names = [field.name() for field in layer.fields()]
            for key, label in self._LABELS.items():
                combo = QComboBox()
                combo.addItem("— не копировать —", "")
                for name in names:
                    combo.addItem(name, name)
                index = combo.findData(current.get(key, ""))
                combo.setCurrentIndex(max(0, index))
                form.addRow(label, combo)
                self._combos[(role, key)] = combo
            layout.addWidget(group)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # Vozvrashchaet vybrannye sopostavleniya obeih rolei.
    def mappings(self) -> dict[str, dict[str, str]]:
        return {
            role: {
                key: str(self._combos[(role, key)].currentData() or "")
                for key in self._LABELS
            }
            for role in ("annotation", "hard_negative")
        }


class MLSystemDockWidget(QDockWidget):
    """Svyazyvaet neblokiruyushchii API i persistent staging."""

    session_active_changed = pyqtSignal(bool)

    # Inicializiruet sostoyanie bez setevyh ili proektnyh mutacii.
    def __init__(self, iface, parent: QWidget | None = None) -> None:
        super().__init__("MLSystem2 — псевдоразметка", parent)
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.api = APIClient(self)
        self.settings = load_settings()
        self.api.configure(SERVER_URL, self.settings.request_timeout_ms)
        self.session: ReviewSession | None = None
        self._aoi_geometry: QgsGeometry | None = None
        self._aoi_crs = None
        self._job_id: str | None = None
        self._job_class_name: str | None = None
        self._connecting = False
        self._poll_failures = 0
        self._removing_candidate_layer = False
        self._previous_map_tool = None
        self._aoi_tool = AOIPolygonMapTool(self.canvas)
        self._aoi_tool.captured.connect(self._aoi_captured)
        self._candidate_highlight = QgsRubberBand(self.canvas, Qgis.GeometryType.Polygon)
        self._candidate_highlight.setStrokeColor(QColor(220, 0, 255, 255))
        self._candidate_highlight.setFillColor(QColor(255, 235, 0, 80))
        self._candidate_highlight.setWidth(3)
        self._candidate_highlight.hide()
        self._poll_timer = QTimer(self)
        self._poll_timer.setSingleShot(True)
        self._poll_timer.timeout.connect(self._poll_job)
        self.api.succeeded.connect(self._api_succeeded)
        self.api.failed.connect(self._api_failed)
        QgsProject.instance().layersRemoved.connect(self._layers_removed)
        self._build_ui()
        self._load_settings_to_ui()
        self.refresh_layers()
        self._update_review_ui()

    # Stroit kompaktnyi UI programmnymi sredstvami Qt.
    def _build_ui(self) -> None:
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(6, 6, 6, 6)

        recognition = QGroupBox("Распознавание")
        recognition_layout = QGridLayout(recognition)
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._class_changed)
        self.model_label = QLabel("Модель: —")
        draw_aoi = QPushButton("Нарисовать")
        draw_aoi.clicked.connect(self._draw_aoi)
        self.aoi_label = QLabel("AOI не задана")
        self.aoi_label.setWordWrap(True)
        self.start_button = QPushButton("Запустить распознавание")
        self.cancel_button = QPushButton("Отменить задание")
        self.cancel_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_job)
        self.cancel_button.clicked.connect(self._cancel_job)
        self.status_label = QLabel("Статус: —")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        recognition_layout.addWidget(QLabel("Класс"), 0, 0)
        recognition_layout.addWidget(self.class_combo, 0, 1, 1, 2)
        recognition_layout.addWidget(self.model_label, 1, 0, 1, 3)
        recognition_layout.addWidget(draw_aoi, 2, 0, 1, 3)
        recognition_layout.addWidget(self.aoi_label, 3, 0, 1, 3)
        recognition_layout.addWidget(self.start_button, 4, 0, 1, 2)
        recognition_layout.addWidget(self.cancel_button, 4, 2)
        recognition_layout.addWidget(self.status_label, 5, 0, 1, 3)
        recognition_layout.addWidget(self.progress, 6, 0, 1, 3)
        recognition_layout.addWidget(self.warning_label, 7, 0, 1, 3)
        root.addWidget(recognition)

        layers_group = QGroupBox("Рабочие слои")
        layers_form = QFormLayout(layers_group)
        self.annotation_layer_combo = QComboBox()
        self.hard_layer_combo = QComboBox()
        layer_buttons = QHBoxLayout()
        refresh_layers = QPushButton("Обновить")
        map_fields = QPushButton("Сопоставить поля")
        refresh_layers.clicked.connect(self.refresh_layers)
        map_fields.clicked.connect(self._map_fields)
        layer_buttons.addWidget(refresh_layers)
        layer_buttons.addWidget(map_fields)
        layers_form.addRow("Разметка", self.annotation_layer_combo)
        layers_form.addRow("Hard negative", self.hard_layer_combo)
        layers_form.addRow(layer_buttons)
        root.addWidget(layers_group)

        review = QGroupBox("Проверка кандидатов")
        review_layout = QGridLayout(review)
        self.filter_combo = QComboBox()
        for text, value in (
            ("Только непроверенные", "new"),
            ("Все", "all"),
            ("Принятые", "annotation"),
            ("Hard negative", "hard_negative"),
            ("Отклонённые", "discarded"),
            ("Выгруженные", "exported"),
        ):
            self.filter_combo.addItem(text, value)
        self.sort_combo = QComboBox()
        for text, value in (
            ("Исходный порядок", "original"),
            ("Confidence ↓", "confidence_desc"),
            ("Площадь ↓", "area_desc"),
        ):
            self.sort_combo.addItem(text, value)
        self.filter_combo.currentIndexChanged.connect(self._review_filter_changed)
        self.sort_combo.currentIndexChanged.connect(self._review_sort_changed)
        self.counter_label = QLabel("0 из 0; осталось 0")
        self.candidate_label = QLabel("Текущий объект: —")
        self.candidate_label.setWordWrap(True)
        previous_button = QPushButton("Предыдущий")
        next_button = QPushButton("Следующий")
        zoom_button = QPushButton("Приблизить")
        previous_button.clicked.connect(self.previous_candidate)
        next_button.clicked.connect(self.next_candidate)
        zoom_button.clicked.connect(self.zoom_candidate)
        annotation_button = QPushButton("1 — В разметку")
        hard_button = QPushButton("2 — Hard negative")
        discard_button = QPushButton("3 — Выкинуть")
        split_button = QPushButton("S — Разбить")
        annotation_button.clicked.connect(lambda: self.classify("annotation"))
        hard_button.clicked.connect(lambda: self.classify("hard_negative"))
        discard_button.clicked.connect(lambda: self.classify("discarded"))
        split_button.clicked.connect(self.split_candidate)
        undo_button = QPushButton("Отменить")
        redo_button = QPushButton("Повторить")
        undo_button.clicked.connect(self.undo)
        redo_button.clicked.connect(self.redo)
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(1.0, 1_000_000_000_000.0)
        self.max_area.setDecimals(1)
        self.max_area.setSuffix(" м²")
        self.min_area = QDoubleSpinBox()
        self.min_area.setRange(0.0, 1_000_000_000.0)
        self.min_area.setDecimals(1)
        self.min_area.setSuffix(" м²")
        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setDecimals(3)
        self.min_confidence.setSingleStep(0.05)
        self.min_area.valueChanged.connect(self._review_thresholds_changed)
        self.min_confidence.valueChanged.connect(self._review_thresholds_changed)
        self.auto_split = QCheckBox("Автоматически разбивать после загрузки")
        self.apply_button = QPushButton("Применить результаты в рабочие слои")
        self.apply_button.clicked.connect(self._apply_results)
        self.session_path_label = QLabel("Сессия: —")
        self.session_path_label.setWordWrap(True)
        review_layout.addWidget(QLabel("Фильтр"), 0, 0)
        review_layout.addWidget(self.filter_combo, 0, 1, 1, 2)
        review_layout.addWidget(QLabel("Сортировка"), 1, 0)
        review_layout.addWidget(self.sort_combo, 1, 1, 1, 2)
        review_layout.addWidget(self.counter_label, 2, 0, 1, 3)
        review_layout.addWidget(self.candidate_label, 3, 0, 1, 3)
        review_layout.addWidget(previous_button, 4, 0)
        review_layout.addWidget(next_button, 4, 1)
        review_layout.addWidget(zoom_button, 4, 2)
        review_layout.addWidget(annotation_button, 5, 0)
        review_layout.addWidget(hard_button, 5, 1)
        review_layout.addWidget(discard_button, 5, 2)
        review_layout.addWidget(split_button, 6, 0)
        review_layout.addWidget(undo_button, 6, 1)
        review_layout.addWidget(redo_button, 6, 2)
        review_layout.addWidget(QLabel("Макс. часть"), 7, 0)
        review_layout.addWidget(self.max_area, 7, 1, 1, 2)
        review_layout.addWidget(QLabel("Мин. площадь/часть"), 8, 0)
        review_layout.addWidget(self.min_area, 8, 1, 1, 2)
        review_layout.addWidget(QLabel("Мин. уверенность"), 9, 0)
        review_layout.addWidget(self.min_confidence, 9, 1, 1, 2)
        review_layout.addWidget(self.auto_split, 10, 0, 1, 3)
        review_layout.addWidget(self.apply_button, 11, 0, 1, 3)
        review_layout.addWidget(self.session_path_label, 12, 0, 1, 3)
        root.addWidget(review)
        root.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        self.setWidget(scroll)
        self.setMinimumWidth(360)

    # Perenosit sohranennye nastroiki v kontroly.
    def _load_settings_to_ui(self) -> None:
        self.max_area.setValue(self.settings.max_part_area_m2)
        self.min_area.setValue(self.settings.min_part_area_m2)
        self.min_confidence.setValue(self.settings.min_confidence)
        self.auto_split.setChecked(self.settings.auto_split)

    # Sohranyaet tekushchie kontroly i konfiguriruet API-klient.
    def save_current_settings(self) -> None:
        self.settings = replace(
            self.settings,
            max_part_area_m2=self.max_area.value(),
            min_part_area_m2=self.min_area.value(),
            min_confidence=self.min_confidence.value(),
            auto_split=self.auto_split.isChecked(),
        )
        save_settings(self.settings)
        self.api.configure(SERVER_URL, self.settings.request_timeout_ms)

    def refresh_layers(self) -> None:
        """Obnovit spiski, ne menyaya aktivnyi sloi QGIS."""

        previous = {
            "annotation": self.annotation_layer_combo.currentData(),
            "hard": self.hard_layer_combo.currentData(),
        }
        layers = [layer for layer in polygon_layers() if self.session is None or layer.id() != self.session.layer.id()]
        for combo, key in (
            (self.annotation_layer_combo, "annotation"),
            (self.hard_layer_combo, "hard"),
        ):
            combo.clear()
            combo.addItem("— не выбран —", "")
            for layer in layers:
                combo.addItem(layer.name(), layer.id())
            index = combo.findData(previous[key]) if previous[key] else -1
            if index >= 0:
                combo.setCurrentIndex(index)
            elif key == "hard":
                matches = [layer for layer in layers if _is_hard_negative_layer(layer)]
                if len(matches) == 1:
                    combo.setCurrentIndex(combo.findData(matches[0].id()))
            elif key == "annotation":
                matches = [layer for layer in layers if not _is_hard_negative_layer(layer)]
                if len(matches) == 1:
                    combo.setCurrentIndex(combo.findData(matches[0].id()))

    # Выполняет вход и после него загружает доступные классы.
    def _request_classes(self) -> None:
        if self._connecting:
            return
        self.save_current_settings()
        password = load_connection_password()
        if not password:
            self._show_error("В локальном профиле QGIS не настроен пароль MLSystem.")
            return
        self._connecting = True
        self.status_label.setText("Статус: вход…")
        self.api.login(SERVER_USERNAME, password)

    def _ensure_connected(self) -> None:
        """Автоматически подключиться, если список классов ещё не загружен."""

        if not self._connecting and self.class_combo.count() == 0:
            self._request_classes()

    # Pokazyvaet versiyu modeli vybrannogo klassa.
    def _class_changed(self) -> None:
        item = self.class_combo.currentData()
        self.model_label.setText(
            f"Модель: {item.get('model_name')} / {item.get('model_version')}"
            if isinstance(item, dict)
            else "Модель: —"
        )

    # Aktiviruet instrument risovaniya bez smeny aktivnogo sloya.
    def _draw_aoi(self) -> None:
        current_tool = self.canvas.mapTool()
        if current_tool is not self._aoi_tool:
            self._previous_map_tool = current_tool
        self._aoi_tool.reset()
        self.canvas.setMapTool(self._aoi_tool)
        self.status_label.setText("Статус: рисуйте левой кнопкой, завершите правой")

    # Prinimaet rezultat map tool i vosstanavlivaet prezhnii instrument.
    def _aoi_captured(self, geometry: QgsGeometry, crs) -> None:
        self._set_aoi(QgsGeometry(geometry), crs)
        QTimer.singleShot(0, self._restore_previous_map_tool)

    def _restore_previous_map_tool(self) -> None:
        previous_tool = self._previous_map_tool
        self._previous_map_tool = None
        if previous_tool is not None and self.canvas.mapTool() is self._aoi_tool:
            self.canvas.setMapTool(previous_tool)

    # Proveriaet i sohranyaet itogovuyu geometriyu AOI.
    def _set_aoi(self, geometry: QgsGeometry, crs) -> None:
        geometry = geometry.makeValid()
        if geometry.isNull() or geometry.isEmpty() or not crs.isValid():
            self._show_error("AOI или её CRS некорректна.")
            return
        if geometry.type() != Qgis.GeometryType.Polygon:
            self._show_error("AOI должна быть Polygon или MultiPolygon.")
            return
        calculator = QgsDistanceArea()
        calculator.setSourceCrs(crs, QgsProject.instance().transformContext())
        ellipsoid = QgsProject.instance().ellipsoid()
        calculator.setEllipsoid(ellipsoid if ellipsoid and ellipsoid != "NONE" else "WGS84")
        try:
            area_m2 = abs(float(calculator.measureArea(geometry)))
        except QgsCsException as exc:
            self._show_error(f"Не удалось рассчитать площадь AOI: {exc}")
            return
        self._aoi_geometry = geometry
        self._aoi_crs = crs
        self.aoi_label.setText(
            f"AOI задана; площадь: {_format_area(area_m2)}; "
            f"CRS: {crs.authid() or crs.toWkt()[:40]}"
        )

    # Otpravlyaet tolko class_id, AOI i ee CRS.
    def _start_job(self) -> None:
        self.save_current_settings()
        class_info = self.class_combo.currentData()
        if not isinstance(class_info, dict):
            self._show_error("Выберите класс распознавания.")
            return
        if self._aoi_geometry is None or self._aoi_crs is None:
            last_capture = self._aoi_tool.last_capture()
            if last_capture is not None:
                self._set_aoi(*last_capture)
        if self._aoi_geometry is None or self._aoi_crs is None:
            self._show_error("Сначала задайте AOI.")
            return
        try:
            aoi = json.loads(self._aoi_geometry.asJson())
        except json.JSONDecodeError as exc:
            self._show_error(f"Не удалось сформировать GeoJSON AOI: {exc}")
            return
        if self.session is not None:
            self.close_session(remove_layer=True)
        self._remove_stale_candidate_layers()
        self.refresh_layers()
        self._job_id = None
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._job_class_name = str(class_info.get("display_name") or "").strip() or None
        self.status_label.setText("Статус: создание задания…")
        self.warning_label.clear()
        self.api.create_job(
            str(class_info["class_id"]),
            aoi,
            self._aoi_crs.authid() or self._aoi_crs.toWkt(),
        )

    # Ostanavlivaet polling i zaprashivaet servernuyu otmenu.
    def _cancel_job(self) -> None:
        self._poll_timer.stop()
        if self._job_id:
            self.api.cancel_job(self._job_id)
        else:
            self.api.abort_all()
            self._job_finished()

    # Zaprashivaet odin snapshot tekushchego zadaniya.
    def _poll_job(self) -> None:
        if self._job_id:
            self.api.get_job(self._job_id)

    # Marshrutiziruet uspeshnyi asinkhronnyi otvet.
    def _api_succeeded(self, operation: str, payload: object) -> None:
        if operation == "login":
            self.status_label.setText("Статус: вход выполнен; загрузка классов…")
            self.api.get_classes()
        elif operation == "classes":
            self._connecting = False
            self.class_combo.clear()
            classes = payload.get("classes", []) if isinstance(payload, dict) else []
            for item in classes:
                self.class_combo.addItem(str(item.get("display_name") or item.get("class_id")), item)
            self.status_label.setText(f"Статус: подключено; классов: {len(classes)}")
        elif operation == "create_job" and isinstance(payload, dict):
            self._job_id = str(payload["job_id"])
            self.status_label.setText("Статус: в очереди")
            self._schedule_poll(0)
        elif operation == "job_status" and isinstance(payload, dict):
            self._poll_failures = 0
            status = str(payload.get("status") or "")
            stage = str(payload.get("current_stage") or "")
            self.status_label.setText(
                f"Статус: {_status_text(status)}; этап: {_stage_text(stage)}"
            )
            progress = payload.get("progress")
            if progress is None:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 100)
                self.progress.setValue(int(float(progress)))
            warnings = payload.get("warnings") or []
            coverage = payload.get("coverage_percent")
            messages = [str(item) for item in warnings]
            source_image_ids = payload.get("source_image_ids") or []
            if isinstance(source_image_ids, list) and source_image_ids:
                messages.append(f"Выбрано снимков: {len(source_image_ids)}")
            if coverage is not None:
                messages.append(f"Покрытие AOI: {float(coverage):.2f}%")
            self.warning_label.setText("\n".join(messages))
            if status == "succeeded":
                self.api.get_result(self._job_id or "")
            elif status in {"failed", "cancelled"}:
                error = payload.get("error")
                if isinstance(error, dict):
                    details = error.get("details")
                    failed_sources = (
                        details.get("failed_source_image_ids")
                        if isinstance(details, dict)
                        else None
                    )
                    if isinstance(failed_sources, list) and failed_sources:
                        messages.append(
                            "Не обработаны снимки: "
                            + ", ".join(str(item) for item in failed_sources)
                        )
                        self.warning_label.setText("\n".join(messages))
                    self._show_error(str(error.get("message") or "Задание завершилось с ошибкой."))
                self._job_finished()
            else:
                self._schedule_poll(self.settings.poll_interval_ms)
        elif operation == "job_result":
            try:
                validated = validate_feature_collection(payload)
                if self._job_class_name:
                    for feature in validated["features"]:
                        feature["properties"].setdefault("class_name", self._job_class_name)
                session = ReviewSession.from_geojson(
                    validated,
                    self._job_id or "unknown",
                    session_directory(),
                )
                self._set_session(session, reset_filter=True)
                if self.auto_split.isChecked():
                    session.split_large_candidates(self.max_area.value(), self.min_area.value())
                count = session.counts()["total"]
                job_short_id = (self._job_id or "unknown")[:8]
                self.status_label.setText(
                    f"Статус: результат задания {job_short_id} загружен; объектов: {count}"
                )
                if count == 0:
                    self.warning_label.setText("Сервер не нашёл объектов в AOI.")
            except (PluginContractError, ReviewSessionError) as exc:
                self._show_error(str(exc))
            finally:
                self._job_finished()
        elif operation == "cancel_job":
            self.status_label.setText("Статус: отменено")
            self._job_finished()

    # Pokazyvaet oshibku ili planiruet povtor vremennogo sboya.
    def _api_failed(
        self,
        operation: str,
        code: str,
        message: str,
        status: int,
        transient: bool,
    ) -> None:
        QgsMessageLog.logMessage(
            f"Операция {operation}: {code}; HTTP {status}",
            "MLSystem2",
            Qgis.MessageLevel.Warning,
        )
        if operation == "job_status" and transient and self._job_id:
            self._poll_failures += 1
            delay = min(15_000, self.settings.poll_interval_ms * (2 ** min(3, self._poll_failures)))
            self.status_label.setText(f"Статус: временная ошибка сети; повтор через {delay // 1000} с")
            self._schedule_poll(delay)
            return
        self._show_error(f"{message} [{code}]")
        if operation in {"login", "classes"}:
            self._connecting = False
        if code == "SESSION_EXPIRED":
            self.class_combo.clear()
            self._connecting = False
            QTimer.singleShot(0, self._ensure_connected)
        if operation in {"create_job", "job_result", "cancel_job"}:
            self._job_finished()

    # Planiruet sleduyushchii polling cherez event loop.
    def _schedule_poll(self, delay_ms: int) -> None:
        self._poll_timer.start(max(0, delay_ms))

    # Vozvrashchaet knopki v sostoyanie bez aktivnogo job.
    def _job_finished(self) -> None:
        self._poll_timer.stop()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    # Aktiviruet novuyu persistent review session.
    def _set_session(self, session: ReviewSession, *, reset_filter: bool = False) -> None:
        if self.session is not None:
            self.close_session(remove_layer=True)
        self._remove_stale_candidate_layers(except_layer_id=session.layer_id)
        if reset_filter:
            new_filter_index = self.filter_combo.findData("new")
            if new_filter_index >= 0:
                self.filter_combo.setCurrentIndex(new_filter_index)
        self.session = session
        session.changed.connect(self._update_review_ui)
        session.current_changed.connect(self._highlight_current_candidate)
        session.set_filter(str(self.filter_combo.currentData()))
        session.set_sort(str(self.sort_combo.currentData()))
        session.set_thresholds(self.min_area.value(), self.min_confidence.value())
        self.session_path_label.setText(f"Сессия: {session.path}")
        self.refresh_layers()
        self._update_review_ui()
        self._highlight_current_candidate(session.current_feature())
        self.session_active_changed.emit(self.isVisible())

    def _remove_stale_candidate_layers(self, *, except_layer_id: str | None = None) -> None:
        """Убрать с карты старые кандидаты, не удаляя их GeoPackage."""

        project = QgsProject.instance()
        layer_ids = [
            layer.id()
            for layer in project.mapLayers().values()
            if layer.id() != except_layer_id and is_review_session_layer(layer)
        ]
        if not layer_ids:
            return
        self._removing_candidate_layer = True
        try:
            project.removeMapLayers(layer_ids)
        finally:
            self._removing_candidate_layer = False

    # Zakryvaet tekushchuyu sessiyu bez poteri GeoPackage.
    def close_session(self, *, remove_layer: bool) -> None:
        if self.session is None:
            return
        session = self.session
        self.session = None
        self._removing_candidate_layer = True
        try:
            session.close(remove_layer=remove_layer)
        finally:
            self._removing_candidate_layer = False
        self._highlight_current_candidate(None)
        self.session_active_changed.emit(False)
        self._update_review_ui()

    # Deaktiviruet sessiyu pri vneshnem udalenii sloya kandidatov.
    def _layers_removed(self, layer_ids: list[str]) -> None:
        if (
            self._removing_candidate_layer
            or self.session is None
            or self.session.layer_id not in layer_ids
        ):
            return
        self.session.undo_stack.clear()
        self.session = None
        self._highlight_current_candidate(None)
        self.session_active_changed.emit(False)
        self._update_review_ui()
        self._show_error("Слой кандидатов удалён во время сессии.")

    # Peredaet filtr v model sessii.
    def _review_filter_changed(self) -> None:
        if self.session:
            self.session.set_filter(str(self.filter_combo.currentData()))

    # Peredaet sortirovku v model sessii.
    def _review_sort_changed(self) -> None:
        if self.session:
            self.session.set_sort(str(self.sort_combo.currentData()))

    def _review_thresholds_changed(self) -> None:
        """Немедленно применить пороги к очереди и слою кандидатов."""

        if self.session:
            self.session.set_thresholds(self.min_area.value(), self.min_confidence.value())

    def _highlight_current_candidate(self, feature) -> None:
        """Подсветить текущий кандидат поверх карты независимо от масштаба."""

        self._candidate_highlight.reset(Qgis.GeometryType.Polygon)
        if self.session is None or feature is None or not feature.isValid():
            self._candidate_highlight.hide()
            return
        self._candidate_highlight.setToGeometry(feature.geometry(), self.session.layer)
        self._candidate_highlight.show()

    # Obnovlyaet schetchiki i atributy tekushchego kandidata.
    def _update_review_ui(self) -> None:
        if self.session is None:
            self.counter_label.setText("0 из 0; осталось 0")
            self.candidate_label.setText("Текущий объект: —")
            return
        position, visible = self.session.position()
        counts = self.session.counts()
        if visible == 0 and counts["total"] > 0:
            self.counter_label.setText(
                f"0 отображается; скрыто фильтрами: {counts['total']}; "
                f"осталось {counts['new']}; всего {counts['total']}"
            )
        else:
            self.counter_label.setText(
                f"{position} из {visible}; осталось {counts['new']}; всего {counts['total']}"
            )
        feature = self.session.current_feature()
        if feature is None:
            if counts["total"] > 0:
                self.candidate_label.setText(
                    "Нет объектов при текущем отборе.\n"
                    f"Статус: «{self.filter_combo.currentText()}»; "
                    f"площадь ≥ {_format_area(self.min_area.value())}; "
                    f"уверенность ≥ {self.min_confidence.value():.3f}."
                )
            else:
                self.candidate_label.setText("Текущий объект: —")
            return
        confidence = feature["confidence"]
        confidence_text = "—" if confidence is None else f"{float(confidence):.3f}"
        self.candidate_label.setText(
            f"ID: {feature['candidate_id']}\n"
            f"Класс: {feature['class_id']}; версия: {feature['model_version']}\n"
            f"Confidence: {confidence_text}; площадь: {float(feature['area_m2'] or 0):.1f} м²"
        )

    # Klassificiruet tekushchii obekt cherez edinyi undo stack.
    def classify(self, status: str) -> None:
        self._session_call(lambda: self.session.classify(status))

    # Perehodit k sleduyushchemu kandidatu.
    def next_candidate(self) -> None:
        self._session_call(lambda: self.session.next())

    # Perehodit k predydushchemu kandidatu.
    def previous_candidate(self) -> None:
        self._session_call(lambda: self.session.previous())

    # Priblizhaet canvas bez smeny aktivnogo sloya.
    def zoom_candidate(self) -> None:
        if self.session is None or self.session.current_feature() is None:
            return
        extent = self.session.current_feature().geometry().boundingBox()
        transform = QgsCoordinateTransform(
            self.session.layer.crs(),
            self.canvas.mapSettings().destinationCrs(),
            QgsProject.instance(),
        )
        try:
            extent = transform.transformBoundingBox(extent)
        except QgsCsException as exc:
            self._show_error(f"Не удалось преобразовать охват кандидата: {exc}")
            return
        extent.scale(1.3)
        self.canvas.setExtent(extent)
        self.canvas.refresh()

    # Razbivaet tekushchii obekt po nastroyennym porogam.
    def split_candidate(self) -> None:
        self._session_call(
            lambda: self.session.split_current(self.max_area.value(), self.min_area.value())
        )

    # Otmenyaet odno logicheskoe review-deistvie.
    def undo(self) -> None:
        if self.session and self.session.undo_stack.canUndo():
            self.session.undo_stack.undo()

    # Povtoryaet odno otmenennoe review-deistvie.
    def redo(self) -> None:
        if self.session and self.session.undo_stack.canRedo():
            self.session.undo_stack.redo()

    # Otkryvaet nastroiku polei dlya yavno vybrannyh sloev.
    def _map_fields(self) -> None:
        annotation = self._layer_from_combo(self.annotation_layer_combo)
        hard = self._layer_from_combo(self.hard_layer_combo)
        if annotation is None or hard is None or annotation.id() == hard.id():
            self._show_error("Явно выберите два разных рабочих слоя.")
            return
        dialog = FieldMappingDialog(annotation, hard, self)
        if dialog.exec() == _accepted_dialog_code():
            for role, mapping in dialog.mappings().items():
                save_field_mapping(role, mapping)

    # Atomarno perenosit raspredelennye obekty v edit buffers.
    def _apply_results(self) -> None:
        if self.session is None:
            self._show_error("Сначала загрузите сессию кандидатов.")
            return
        candidate_layer = QgsProject.instance().mapLayer(self.session.layer_id)
        if not isinstance(candidate_layer, QgsVectorLayer):
            self._show_error("Слой кандидатов удалён во время сессии.")
            return
        annotation = self._layer_from_combo(self.annotation_layer_combo)
        hard = self._layer_from_combo(self.hard_layer_combo)
        try:
            result = apply_reviewed_candidates(
                candidate_layer,
                annotation,
                hard,
                load_field_mapping("annotation", annotation),
                load_field_mapping("hard_negative", hard),
            )
        except LayerOperationError as exc:
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QgsMessageLog.logMessage(
                f"Непредвиденная ошибка применения: {exc!r}",
                "MLSystem2",
                Qgis.MessageLevel.Critical,
            )
            self._show_error("Не удалось применить результаты. Подробности записаны в журнал QGIS.")
            return
        self.session.undo_stack.clear()
        self._update_review_ui()
        QMessageBox.information(
            self,
            "MLSystem2",
            f"Добавлено: {result.added}; уже существовало: {result.existing}.\n"
            "Правки целевых слоёв не сохранены: проверьте их и сохраните штатной командой QGIS.",
        )

    # Razreshaet ID combo tolko cherez QgsProject.
    def _layer_from_combo(self, combo: QComboBox) -> QgsVectorLayer | None:
        layer_id = str(combo.currentData() or "")
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
        return layer if isinstance(layer, QgsVectorLayer) else None

    # Edinoobrazno obrabatyvaet domennye oshibki sessii.
    def _session_call(self, action) -> None:
        if self.session is None:
            return
        try:
            action()
        except ReviewSessionError as exc:
            self._show_error(str(exc))

    # Pokazyvaet kratkuyu oshibku bez sekretnyh dannyh.
    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Ошибка: {message}")
        self.iface.messageBar().pushMessage(
            "MLSystem2",
            message,
            level=Qgis.MessageLevel.Critical,
            duration=8,
        )

    # Vklyuchaet hotkeys tolko pri vidimoi aktivnoi sessii.
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._ensure_connected)
        if self.session is not None:
            self._highlight_current_candidate(self.session.current_feature())
        self.session_active_changed.emit(self.session is not None)

    # Vozvrashchaet globalnye hotkeys shtatnomu QGIS.
    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._candidate_highlight.hide()
        self.session_active_changed.emit(False)

    # Sohranyaet nastroiki i ostanavlivaet fonovye zaprosy.
    def closeEvent(self, event) -> None:  # noqa: N802
        self.save_current_settings()
        self._poll_timer.stop()
        self.api.abort_all()
        self.session_active_changed.emit(False)
        super().closeEvent(event)
__all__ = ["MLSystemDockWidget"]
