from __future__ import annotations
from . import base

__all__ = ["__version__", "__ceres_version__", "_limap", "base"]

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

import importlib


def __getattr__(name: str):
    if name == "_limap":
        return importlib.import_module("._limap", __name__)
    if name == "__version__":
        return getattr(__getattr__("_limap"), "__version__")
    if name == "__ceres_version__":
        return getattr(__getattr__("_limap"), "__ceres_version__")
    raise AttributeError(name)
