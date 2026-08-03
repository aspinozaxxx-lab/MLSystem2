"""Tochka vhoda QGIS-plagina MLSystem2."""

from __future__ import annotations


def classFactory(iface):  # noqa: N802
    """Sozdat ekzemplyar plagina dlya QGIS."""

    from .plugin import MLSystemPlugin

    return MLSystemPlugin(iface)


__all__ = ["classFactory"]
