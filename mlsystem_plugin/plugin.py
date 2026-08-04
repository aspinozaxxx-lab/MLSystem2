"""Zhiznennyi cikl QGIS-plagina MLSystem2."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QKeySequence
try:
    from qgis.PyQt.QtGui import QAction, QShortcut
except ImportError:
    from qgis.PyQt.QtWidgets import QAction, QShortcut

from .dock_widget import MLSystemDockWidget


# Vozvrashchaet sovmestimuyu oblast dock dlya Qt5/Qt6.
def _dock_area(name: str):
    enum = getattr(Qt, "DockWidgetArea", Qt)
    return getattr(enum, name)


# Vozvrashchaet sovmestimyi kontekst shortcut dlya Qt5/Qt6.
def _shortcut_context(name: str):
    enum = getattr(Qt, "ShortcutContext", Qt)
    return getattr(enum, name)


class MLSystemPlugin:
    """Registriruet dock-panel i globalnye tolko dlya sessii hotkeys."""

    MENU = "&MLSystem2"

    # Sohranyaet QGIS iface bez pobochnyh deistvii.
    def __init__(self, iface) -> None:
        self.iface = iface
        self.dock: MLSystemDockWidget | None = None
        self.action: QAction | None = None
        self._shortcuts: list[QShortcut] = []

    # Registriruet panel, menu i otklyuchennye hotkeys.
    def initGui(self) -> None:  # noqa: N802
        self.dock = MLSystemDockWidget(self.iface, self.iface.mainWindow())
        self.iface.addDockWidget(_dock_area("RightDockWidgetArea"), self.dock)
        self.action = QAction("Распознавание объектов MLSystem2", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.setChecked(True)
        self.action.triggered.connect(self._toggle_dock)
        self.dock.visibilityChanged.connect(self.action.setChecked)
        self.iface.addPluginToVectorMenu(self.MENU, self.action)
        self._create_shortcuts()
        self.dock.session_active_changed.connect(self._set_shortcuts_enabled)

    # Polnostyu osvobozhdaet UI, hotkeys i vnutrennii sloi.
    def unload(self) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self._shortcuts.clear()
        if self.action is not None:
            self.iface.removePluginVectorMenu(self.MENU, self.action)
            self.action.deleteLater()
            self.action = None
        if self.dock is not None:
            self.dock.save_current_settings()
            if self.dock.session is not None:
                self.dock.close_session(remove_layer=True)
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None

    # Sinhroniziruet vidimost paneli s punktom menu.
    def _toggle_dock(self, visible: bool) -> None:
        if self.dock is not None:
            self.dock.setVisible(visible)
            if visible:
                self.dock.raise_()

    # Sozdaet edinyi nabor application shortcuts.
    def _create_shortcuts(self) -> None:
        assert self.dock is not None
        bindings = (
            ("N", self.dock.next_candidate),
            ("Space", self.dock.next_candidate),
            ("P", self.dock.previous_candidate),
            ("S", self.dock.split_candidate),
        )
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self.iface.mainWindow())
            shortcut.setContext(_shortcut_context("ApplicationShortcut"))
            shortcut.activated.connect(callback)
            shortcut.setEnabled(False)
            self._shortcuts.append(shortcut)

    # Vklyuchaet hotkeys tolko dlya aktivnoi sessii.
    def _set_shortcuts_enabled(self, enabled: bool) -> None:
        for shortcut in self._shortcuts:
            shortcut.setEnabled(bool(enabled))


__all__ = ["MLSystemPlugin"]
