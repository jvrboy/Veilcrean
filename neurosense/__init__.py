"""Compatibility shim for importing NeuroSense from the repository root.

The Dropbox archive is laid out as a standalone Python project under this
``neurosense/`` directory. When tests are run from the parent repository, this
shim forwards ``import neurosense`` and ``import neurosense.<submodule>`` to the
actual package in ``neurosense/neurosense``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_inner_path = Path(__file__).resolve().parent / "neurosense"
if str(_inner_path) not in __path__:
    __path__.append(str(_inner_path))

_inner = importlib.import_module(".neurosense", __name__)

for _name in getattr(_inner, "__all__", []):
    globals()[_name] = getattr(_inner, _name)

for _submodule in ("brain", "ears", "eyes", "knowledge", "language", "learning", "neurons"):
    sys.modules[f"{__name__}.{_submodule}"] = importlib.import_module(f".neurosense.{_submodule}", __name__)

__all__ = list(getattr(_inner, "__all__", []))
__version__ = getattr(_inner, "__version__", "0.1.0")
