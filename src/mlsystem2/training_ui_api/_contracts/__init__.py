"""Public DTO collection for training UI API."""

from __future__ import annotations

from importlib import import_module

_MODULES = (
    "bootstrap",
    "catalog",
    "common",
    "jobs",
    "results",
    "templates",
)

__all__: list[str] = []

for _module_name in _MODULES:
    _module = import_module(f"{__name__}.{_module_name}")
    for _name in _module.__all__:
        globals()[_name] = getattr(_module, _name)
        __all__.append(_name)

del _module
del _module_name
del _name
