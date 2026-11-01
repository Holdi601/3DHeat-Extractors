"""
Every game the watcher knows, gathered from the adapter modules.

A module adds its adapters by listing them in `ADAPTERS`; this collects them.
An adapter module that fails to import (a dependency missing, a platform it
does not run on) leaves its games out and says why in `broken`, rather than
taking the watcher down with it.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import Adapter, Context

#: id -> adapter
REGISTRY: dict[str, Adapter] = {}
#: module -> why it could not be loaded
broken: dict[str, str] = {}


def load() -> dict[str, Adapter]:
    if REGISTRY:
        return REGISTRY
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name in ("base",) or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as e:  # noqa: BLE001 - one game's problem must not stop the others
            broken[info.name] = f"{type(e).__name__}: {e}"
            continue
        for adapter in getattr(module, "ADAPTERS", ()):
            if adapter.id in REGISTRY:
                raise ValueError(f"two adapters are called {adapter.id!r}")
            REGISTRY[adapter.id] = adapter
    return REGISTRY


__all__ = ["Adapter", "Context", "REGISTRY", "broken", "load"]
