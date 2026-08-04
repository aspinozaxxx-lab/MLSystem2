"""Neblokiruyushchii HTTP-klient AOI API."""

from __future__ import annotations

import json
from typing import Any

from qgis.PyQt.QtCore import QByteArray, QObject, QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtNetwork import (
    QNetworkAccessManager,
    QNetworkCookieJar,
    QNetworkReply,
    QNetworkRequest,
)

from .contracts import build_job_payload


class APIClient(QObject):
    """Vypolnyaet HTTP-zaprosy cherez event loop Qt."""

    succeeded = pyqtSignal(str, object)
    failed = pyqtSignal(str, str, str, int, bool)

    # Inicializiruet edinstvennyi Qt network manager.
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._manager = QNetworkAccessManager(self)
        self._server_url = ""
        self._timeout_ms = 30_000
        self._pending: set[QNetworkReply] = set()

    def configure(self, server_url: str, timeout_ms: int) -> None:
        """Обновить адрес сервера и сетевой тайм-аут."""

        normalized_url = server_url.rstrip("/")
        if normalized_url != self._server_url:
            self._manager.setCookieJar(QNetworkCookieJar(self._manager))
        self._server_url = normalized_url
        self._timeout_ms = max(1_000, int(timeout_ms))

    def login(self, username: str, password: str) -> None:
        """Создать серверную сессию по обычной учётной записи MLSystem."""

        self._request(
            "login",
            "POST",
            "/api/v1/auth/login",
            {"username": username, "password": password},
        )

    # Zaprashivaet dostupnye klassy.
    def get_classes(self, operation: str = "classes") -> None:
        self._request(operation, "GET", "/api/v1/pseudolabel/classes")

    # Sozdaet zadanie s minimalnym telom.
    def create_job(
        self,
        class_id: str,
        aoi: dict[str, Any],
        aoi_crs: str,
        source_id: str | None = None,
    ) -> None:
        self._request(
            "create_job",
            "POST",
            "/api/v1/pseudolabel/jobs",
            build_job_payload(class_id, aoi, aoi_crs, source_id),
        )

    # Zaprashivaet sostoyanie zadaniya.
    def get_job(self, job_id: str) -> None:
        self._request("job_status", "GET", f"/api/v1/pseudolabel/jobs/{job_id}")

    # Zagruzhaet gotovyi GeoJSON.
    def get_result(self, job_id: str) -> None:
        self._request("job_result", "GET", f"/api/v1/pseudolabel/jobs/{job_id}/result")

    # Zaprashivaet bezopasnuyu otmenu zadaniya.
    def cancel_job(self, job_id: str) -> None:
        self._request("cancel_job", "DELETE", f"/api/v1/pseudolabel/jobs/{job_id}")

    def abort_all(self) -> None:
        """Ostanovit vse tekushchie setevye zaprosy."""

        for reply in tuple(self._pending):
            reply.abort()

    # Sozdaet neblokiruyushchii HTTP-zapros.
    def _request(
        self,
        operation: str,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        base_url = QUrl(self._server_url)
        if (
            not self._server_url
            or not base_url.isValid()
            or base_url.scheme() not in {"http", "https"}
            or not base_url.host()
        ):
            self.failed.emit(operation, "INVALID_URL", "Не задан URL сервера.", 0, False)
            return
        request = QNetworkRequest(QUrl(f"{self._server_url}{path}"))
        request.setRawHeader(QByteArray(b"Accept"), QByteArray(b"application/json"))
        request.setRawHeader(QByteArray(b"Content-Type"), QByteArray(b"application/json"))
        data = QByteArray(json.dumps(payload or {}, ensure_ascii=False).encode("utf-8"))
        if method == "GET":
            reply = self._manager.get(request)
        elif method == "POST":
            reply = self._manager.post(request, data)
        elif method == "DELETE":
            reply = self._manager.deleteResource(request)
        else:
            raise ValueError(f"Неподдерживаемый HTTP-метод: {method}.")
        self._pending.add(reply)
        timer = QTimer(reply)
        timer.setSingleShot(True)
        timer.timeout.connect(reply.abort)
        timer.start(self._timeout_ms)
        reply.finished.connect(lambda: self._finished(operation, reply, timer))

    # Разбирает ответ без вывода пароля или cookie в журнал.
    def _finished(self, operation: str, reply: QNetworkReply, timer: QTimer) -> None:
        timer.stop()
        self._pending.discard(reply)
        request_attributes = getattr(QNetworkRequest, "Attribute", QNetworkRequest)
        status_attribute = request_attributes.HttpStatusCodeAttribute
        status = int(reply.attribute(status_attribute) or 0)
        raw = bytes(reply.readAll())
        try:
            payload = json.loads(raw.decode("utf-8-sig")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        network_errors = getattr(QNetworkReply, "NetworkError", QNetworkReply)
        network_error = reply.error() != network_errors.NoError
        reply.deleteLater()
        if not network_error and 200 <= status < 300 and payload is not None:
            self.succeeded.emit(operation, payload)
            return
        code = "NETWORK_ERROR"
        message = "Сервер недоступен или вернул повреждённый ответ."
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
        elif status == 401 and operation == "login":
            code = "AUTH_FAILED"
            message = str(payload.get("detail") or "Неверные имя пользователя или пароль.")
        elif status == 401:
            code = "SESSION_EXPIRED"
            message = "Сеанс истёк. Войдите повторно."
        elif status == 403:
            code = "ACCESS_DENIED"
            message = "Сервер запретил доступ для этой учётной записи."
        elif status:
            code = f"HTTP_{status}"
            message = f"Сервер вернул ошибку HTTP {status}."
        transient = network_error or status in {0, 408, 429, 500, 502, 503, 504}
        self.failed.emit(operation, code, message, status, transient)


__all__ = ["APIClient"]
