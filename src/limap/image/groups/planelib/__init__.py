from .register_plane_detector import (
    get_plane_detector,
    DetectorOptions,
    PxwPlanarOptions,
)
from .utils import convert_plane_mask_to_groups2d

# The visualisation helpers need seaborn, which lives in the `viz` extra.
_LAZY_ATTRS = {
    "visualize_top_components": "viz",
    "visualize_single_component": "viz",
    "visualize_plane_tracks": "viz",
}

__all__ = [
    "get_plane_detector",
    "DetectorOptions",
    "PxwPlanarOptions",
    "convert_plane_mask_to_groups2d",
    "visualize_top_components",
    "visualize_single_component",
    "visualize_plane_tracks",
]


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
