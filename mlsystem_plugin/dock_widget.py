"""Kompaktnaya dock-panel proverki psevdorazmetki."""

from __future__ import annotations

import json
from dataclasses import replace

from qgis.PyQt.QtCore import QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
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
)
from qgis.gui import QgsRubberBand

from .aoi_map_tool import AOIPolygonMapTool
from .api_client import APIClient
from .contracts import PluginContractError, validate_feature_collection
from .review_session import ReviewSession, ReviewSessionError, is_review_session_layer
from .settings import (
    SERVER_URL,
    SERVER_USERNAME,
    load_connection_password,
    load_settings,
    save_settings,
    session_directory,
)


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
        "downloading_imagery": "загрузка внешних снимков",
        "preparing_resolution": "подготовка разрешения",
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




class MLSystemDockWidget(QDockWidget):
    """Svyazyvaet neblokiruyushchii API i persistent staging."""

    session_active_changed = pyqtSignal(bool)

    # Inicializiruet sostoyanie bez setevyh ili proektnyh mutacii.
    def __init__(self, iface, parent: QWidget | None = None) -> None:
        super().__init__("MLSystem2 — распознавание объектов", parent)
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
        self._last_job_progress = 0
        self._removing_candidate_layer = False
        self._previous_map_tool = None
        self._aoi_tool = AOIPolygonMapTool(self.canvas)
        self._aoi_tool.captured.connect(self._aoi_captured)
        self._aoi_tool.cancelled.connect(self._aoi_cancelled)
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
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.model_label = QLabel("Модель: —")
        self.draw_aoi_button = QPushButton("Нарисовать AOI")
        self.use_aoi_button = QPushButton("Использовать эту AOI")
        self.use_aoi_button.setEnabled(False)
        self.draw_aoi_button.clicked.connect(self._draw_aoi)
        self.use_aoi_button.clicked.connect(self._use_aoi)
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
        recognition_layout.addWidget(QLabel("Источник снимков"), 1, 0)
        recognition_layout.addWidget(self.source_combo, 1, 1, 1, 2)
        recognition_layout.addWidget(self.model_label, 2, 0, 1, 3)
        recognition_layout.addWidget(self.draw_aoi_button, 3, 0)
        recognition_layout.addWidget(self.use_aoi_button, 3, 1, 1, 2)
        recognition_layout.addWidget(self.aoi_label, 4, 0, 1, 3)
        recognition_layout.addWidget(self.start_button, 5, 0, 1, 2)
        recognition_layout.addWidget(self.cancel_button, 5, 2)
        recognition_layout.addWidget(self.status_label, 6, 0, 1, 3)
        recognition_layout.addWidget(self.progress, 7, 0, 1, 3)
        recognition_layout.addWidget(self.warning_label, 8, 0, 1, 3)
        root.addWidget(recognition)

        review = QGroupBox("Результаты распознавания")
        review_layout = QGridLayout(review)
        self.sort_combo = QComboBox()
        for text, value in (
            ("Исходный порядок", "original"),
            ("Confidence ↓", "confidence_desc"),
            ("Площадь ↓", "area_desc"),
        ):
            self.sort_combo.addItem(text, value)
        self.sort_combo.currentIndexChanged.connect(self._review_sort_changed)
        self.object_type_combo = QComboBox()
        self.object_type_combo.addItem("Все типы", None)
        self.object_type_combo.currentIndexChanged.connect(self._review_object_type_changed)
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("Один категоризированный слой", "categorized")
        self.display_mode_combo.addItem("Группа представлений по типам", "group")
        self.display_mode_combo.currentIndexChanged.connect(self._review_display_mode_changed)
        self.counter_label = QLabel("0 из 0")
        self.candidate_label = QLabel("Текущий объект: —")
        self.candidate_label.setWordWrap(True)
        previous_button = QPushButton("Предыдущий")
        next_button = QPushButton("Следующий")
        zoom_button = QPushButton("Приблизить")
        previous_button.clicked.connect(self.previous_candidate)
        next_button.clicked.connect(self.next_candidate)
        zoom_button.clicked.connect(self.zoom_candidate)
        split_button = QPushButton("Разбить текущий")
        split_all_button = QPushButton("Разбить все крупные")
        split_button.clicked.connect(self.split_candidate)
        split_all_button.clicked.connect(self.split_all_candidates)
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
        self.export_button = QPushButton("Выгрузить на новый слой")
        self.export_button.clicked.connect(self._export_filtered_layer)
        review_layout.addWidget(QLabel("Сортировка"), 0, 0)
        review_layout.addWidget(self.sort_combo, 0, 1, 1, 2)
        review_layout.addWidget(QLabel("Тип объекта"), 1, 0)
        review_layout.addWidget(self.object_type_combo, 1, 1, 1, 2)
        review_layout.addWidget(QLabel("Отображение"), 2, 0)
        review_layout.addWidget(self.display_mode_combo, 2, 1, 1, 2)
        review_layout.addWidget(self.counter_label, 3, 0, 1, 3)
        review_layout.addWidget(self.candidate_label, 4, 0, 1, 3)
        review_layout.addWidget(previous_button, 5, 0)
        review_layout.addWidget(next_button, 5, 1)
        review_layout.addWidget(zoom_button, 5, 2)
        review_layout.addWidget(split_button, 6, 0, 1, 2)
        review_layout.addWidget(split_all_button, 6, 2)
        review_layout.addWidget(QLabel("Макс. площадь части"), 7, 0)
        review_layout.addWidget(self.max_area, 7, 1, 1, 2)
        review_layout.addWidget(QLabel("Мин. площадь"), 8, 0)
        review_layout.addWidget(self.min_area, 8, 1, 1, 2)
        review_layout.addWidget(QLabel("Мин. уверенность"), 9, 0)
        review_layout.addWidget(self.min_confidence, 9, 1, 1, 2)
        review_layout.addWidget(self.export_button, 10, 0, 1, 3)
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
        self.min_area.setValue(self.settings.min_area_m2)
        self.min_confidence.setValue(self.settings.min_confidence)

    # Sohranyaet tekushchie kontroly i konfiguriruet API-klient.
    def save_current_settings(self) -> None:
        self.settings = replace(
            self.settings,
            max_part_area_m2=self.max_area.value(),
            min_area_m2=self.min_area.value(),
            min_confidence=self.min_confidence.value(),
        )
        save_settings(self.settings)
        self.api.configure(SERVER_URL, self.settings.request_timeout_ms)

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
        if isinstance(item, dict):
            default_source = str(item.get("model_imagery_type") or "")
            source_index = next(
                (
                    index
                    for index in range(self.source_combo.count())
                    if isinstance(self.source_combo.itemData(index), dict)
                    and str(self.source_combo.itemData(index).get("source_id") or "")
                    == default_source
                ),
                -1,
            )
            if source_index >= 0:
                self.source_combo.setCurrentIndex(source_index)
        self._update_model_description()

    def _source_changed(self) -> None:
        self._update_model_description()

    def _update_model_description(self) -> None:
        item = self.class_combo.currentData()
        source = self.source_combo.currentData()
        if not isinstance(item, dict):
            self.model_label.setText("Модель: —")
            return
        imagery_type = str(item.get("model_imagery_type") or "—")
        channels = item.get("input_channels") or "—"
        resolution = item.get("target_resolution_m")
        resolution_text = f"{float(resolution):g} м/пикс." if resolution is not None else "не определено"
        self.model_label.setText(
            f"Модель: {item.get('model_name')} / {item.get('model_version')}; "
            f"{imagery_type}, {channels} кан., {resolution_text}"
        )
        messages = []
        if isinstance(source, dict):
            source_type = str(source.get("imagery_type") or "external_rgb")
            if source_type != imagery_type:
                messages.append("Перекрёстное применение: снимки будут приведены к разрешению модели.")
            if channels == 4 and source_type != "kanopus":
                messages.append("NIR отсутствует: четвёртый канал будет заполнен нулями.")
        self.warning_label.setText("\n".join(messages))

    # Aktiviruet instrument risovaniya bez smeny aktivnogo sloya.
    def _draw_aoi(self) -> None:
        current_tool = self.canvas.mapTool()
        if current_tool is not self._aoi_tool:
            self._previous_map_tool = current_tool
        self._aoi_tool.reset()
        self._aoi_geometry = None
        self._aoi_crs = None
        self.aoi_label.setText("AOI рисуется…")
        self.use_aoi_button.setEnabled(True)
        self.canvas.setMapTool(self._aoi_tool)
        self.status_label.setText(
            "Статус: добавляйте вершины левой кнопкой; правая кнопка сбрасывает контур"
        )

    def _use_aoi(self) -> None:
        """Подтвердить нарисованный полигон отдельной кнопкой панели."""

        if self.canvas.mapTool() is not self._aoi_tool:
            self._show_error("Сначала нажмите «Нарисовать AOI».")
            return
        if not self._aoi_tool.capture_current():
            self._show_error("Для AOI укажите не менее трёх вершин.")

    def _aoi_cancelled(self) -> None:
        """Сбросить AOI, не выключая инструмент рисования."""

        self._aoi_geometry = None
        self._aoi_crs = None
        self.aoi_label.setText("AOI не задана; рисуйте заново")
        self.status_label.setText("Статус: AOI сброшена; начните с первой вершины")

    # Prinimaet rezultat map tool i vosstanavlivaet prezhnii instrument.
    def _aoi_captured(self, geometry: QgsGeometry, crs) -> None:
        if self._set_aoi(QgsGeometry(geometry), crs):
            self.use_aoi_button.setEnabled(False)
            QTimer.singleShot(0, self._restore_previous_map_tool)

    def _restore_previous_map_tool(self) -> None:
        previous_tool = self._previous_map_tool
        self._previous_map_tool = None
        if previous_tool is not None and self.canvas.mapTool() is self._aoi_tool:
            self.canvas.setMapTool(previous_tool)

    # Proveriaet i sohranyaet itogovuyu geometriyu AOI.
    def _set_aoi(self, geometry: QgsGeometry, crs) -> bool:
        geometry = geometry.makeValid()
        if geometry.isNull() or geometry.isEmpty() or not crs.isValid():
            self._show_error("AOI или её CRS некорректна.")
            return False
        if geometry.type() != Qgis.GeometryType.Polygon:
            self._show_error("AOI должна быть Polygon или MultiPolygon.")
            return False
        calculator = QgsDistanceArea()
        calculator.setSourceCrs(crs, QgsProject.instance().transformContext())
        ellipsoid = QgsProject.instance().ellipsoid()
        calculator.setEllipsoid(ellipsoid if ellipsoid and ellipsoid != "NONE" else "WGS84")
        try:
            area_m2 = abs(float(calculator.measureArea(geometry)))
        except QgsCsException as exc:
            self._show_error(f"Не удалось рассчитать площадь AOI: {exc}")
            return False
        self._aoi_geometry = geometry
        self._aoi_crs = crs
        self.aoi_label.setText(
            f"AOI задана; площадь: {_format_area(area_m2)}; "
            f"CRS: {crs.authid() or crs.toWkt()[:40]}"
        )
        return True

    # Otpravlyaet tolko class_id, AOI i ee CRS.
    def _start_job(self) -> None:
        self.save_current_settings()
        class_info = self.class_combo.currentData()
        source_info = self.source_combo.currentData()
        if not isinstance(class_info, dict):
            self._show_error("Выберите класс распознавания.")
            return
        if not isinstance(source_info, dict):
            self._show_error("Выберите источник снимков.")
            return
        if self._aoi_geometry is None or self._aoi_crs is None:
            self._show_error("Нарисуйте область и нажмите «Использовать эту AOI».")
            return
        try:
            aoi = json.loads(self._aoi_geometry.asJson())
        except json.JSONDecodeError as exc:
            self._show_error(f"Не удалось сформировать GeoJSON AOI: {exc}")
            return
        if self.session is not None:
            self.close_session(remove_layer=True)
        self._remove_stale_candidate_layers()
        self._job_id = None
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._last_job_progress = 0
        self.progress.setRange(0, 0)
        self._job_class_name = str(class_info.get("display_name") or "").strip() or None
        self.status_label.setText("Статус: создание задания…")
        self.warning_label.clear()
        self.api.create_job(
            str(class_info["class_id"]),
            aoi,
            self._aoi_crs.authid() or self._aoi_crs.toWkt(),
            str(source_info["source_id"]),
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
            self.source_combo.clear()
            classes = payload.get("classes", []) if isinstance(payload, dict) else []
            sources = payload.get("sources", []) if isinstance(payload, dict) else []
            for item in sources:
                if not item.get("available", True):
                    continue
                self.source_combo.addItem(
                    str(item.get("display_name") or item.get("source_id")),
                    item,
                )
            for item in classes:
                self.class_combo.addItem(str(item.get("display_name") or item.get("class_id")), item)
            self._class_changed()
            self.status_label.setText(
                f"Статус: подключено; классов: {len(classes)}; источников: {self.source_combo.count()}"
            )
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
                self._last_job_progress = min(100, max(0, int(float(progress))))
                self.progress.setRange(0, 100)
                self.progress.setValue(self._last_job_progress)
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
                self._set_session(session)
                count = session.counts()["total"]
                job_short_id = (self._job_id or "unknown")[:8]
                self.status_label.setText(
                    f"Статус: результат задания {job_short_id} загружен; объектов: {count}"
                )
                if count == 0:
                    self.warning_label.setText("Сервер не нашёл объектов в AOI.")
            except (PluginContractError, ReviewSessionError) as exc:
                self._show_error(str(exc))
                self._job_finished()
            else:
                self._job_finished(succeeded=True)
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
        if operation in {"create_job", "job_status", "job_result", "cancel_job"}:
            self._job_finished()

    # Planiruet sleduyushchii polling cherez event loop.
    def _schedule_poll(self, delay_ms: int) -> None:
        self._poll_timer.start(max(0, delay_ms))

    # Vozvrashchaet knopki v sostoyanie bez aktivnogo job.
    def _job_finished(self, *, succeeded: bool = False) -> None:
        self._poll_timer.stop()
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if succeeded else self._last_job_progress)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    # Aktiviruet novuyu persistent review session.
    def _set_session(self, session: ReviewSession) -> None:
        if self.session is not None:
            self.close_session(remove_layer=True)
        self._remove_stale_candidate_layers(except_layer_id=session.layer_id)
        self.session = session
        session.changed.connect(self._update_review_ui)
        session.current_changed.connect(self._highlight_current_candidate)
        session.set_sort(str(self.sort_combo.currentData()))
        session.set_thresholds(self.min_area.value(), self.min_confidence.value())
        self.object_type_combo.blockSignals(True)
        self.object_type_combo.clear()
        self.object_type_combo.addItem("Все типы", None)
        for item in session.object_types():
            self.object_type_combo.addItem(str(item["name"]), str(item["slug"]))
        self.object_type_combo.blockSignals(False)
        session.set_object_type_filter(None)
        session.set_display_mode(str(self.display_mode_combo.currentData()))
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
        self.session = None
        self._highlight_current_candidate(None)
        self.session_active_changed.emit(False)
        self._update_review_ui()
        self._show_error("Слой кандидатов удалён во время сессии.")

    # Peredaet sortirovku v model sessii.
    def _review_sort_changed(self) -> None:
        if self.session:
            self.session.set_sort(str(self.sort_combo.currentData()))

    def _review_thresholds_changed(self) -> None:
        """Немедленно применить пороги к очереди и слою кандидатов."""

        if self.session:
            self.session.set_thresholds(self.min_area.value(), self.min_confidence.value())

    def _review_object_type_changed(self) -> None:
        if self.session:
            value = self.object_type_combo.currentData()
            self.session.set_object_type_filter(str(value) if value else None)

    def _review_display_mode_changed(self) -> None:
        if self.session:
            try:
                self.session.set_display_mode(str(self.display_mode_combo.currentData()))
            except ReviewSessionError as exc:
                self._show_error(str(exc))

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
            self.counter_label.setText("0 из 0")
            self.candidate_label.setText("Текущий объект: —")
            return
        position, visible = self.session.position()
        counts = self.session.counts()
        if visible == 0 and counts["total"] > 0:
            self.counter_label.setText(
                f"0 отображается; скрыто фильтрами: {counts['total']}; всего: {counts['total']}"
            )
        else:
            self.counter_label.setText(f"{position} из {visible}; всего: {counts['total']}")
        feature = self.session.current_feature()
        if feature is None:
            if counts["total"] > 0:
                self.candidate_label.setText(
                    "Нет объектов при текущем отборе.\n"
                    f"Площадь ≥ {_format_area(self.min_area.value())}; "
                    f"уверенность ≥ {self.min_confidence.value():.3f}."
                )
            else:
                self.candidate_label.setText("Текущий объект: —")
            return
        confidence = feature["confidence"]
        confidence_text = "—" if confidence is None else f"{float(confidence):.3f}"
        type_name = (
            str(feature["object_type_name"] or "—")
            if self.session.layer.fields().indexOf("object_type_name") >= 0
            else "—"
        )
        self.candidate_label.setText(
            f"ID: {feature['candidate_id']}\n"
            f"Класс: {feature['class_id']}; версия: {feature['model_version']}\n"
            f"Тип: {type_name}\n"
            f"Confidence: {confidence_text}; площадь: {float(feature['area_m2'] or 0):.1f} м²"
        )

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
        self._session_call(lambda: self.session.split_current(self.max_area.value()))

    def split_all_candidates(self) -> None:
        """Разбить все прошедшие фильтры объекты крупнее максимума."""

        if self.session is None:
            return
        try:
            count = self.session.split_large_candidates(self.max_area.value())
        except ReviewSessionError as exc:
            self._show_error(str(exc))
            return
        self.status_label.setText(f"Статус: разбито крупных объектов: {count}")

    def _export_filtered_layer(self) -> None:
        """Выгрузить текущий отбор в отдельный временный слой QGIS."""

        if self.session is None:
            self._show_error("Сначала выполните распознавание.")
            return
        class_name = self._job_class_name or "Результат распознавания"
        try:
            layer = self.session.export_filtered_layer(f"{class_name} — MLSystem2")
        except ReviewSessionError as exc:
            self._show_error(str(exc))
            return
        QMessageBox.information(
            self,
            "MLSystem2",
            f"Создан временный слой «{layer.name()}»: {layer.featureCount()} объектов.\n"
            "Для постоянного хранения сохраните его штатной командой QGIS.",
        )

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
