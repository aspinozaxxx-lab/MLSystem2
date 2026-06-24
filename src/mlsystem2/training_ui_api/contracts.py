"""Backward-compatible public contracts re-export."""

from __future__ import annotations

from . import _contracts

__all__ = list(_contracts.__all__)

for _name in __all__:
    globals()[_name] = getattr(_contracts, _name)

del _name
