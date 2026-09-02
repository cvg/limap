from .specs import PointDetectionOptions, PointMatcherOptions

# superpoint and superglue pull in torch (and hloc), which are optional,
# git-sourced dependencies. Load them on first attribute access so that
# ``import limap.image`` stays cheap and does not require them.
_LAZY_MODULES = frozenset({"superpoint", "superglue"})

__all__ = [
    "superpoint",
    "superglue",
    "PointDetectionOptions",
    "PointMatcherOptions",
]


def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
