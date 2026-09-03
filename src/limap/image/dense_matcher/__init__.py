# register_dense_matcher and specs are cheap; base_dense_matcher, metrics and
# associate import torch, an optional dependency that arrives with the
# git-sourced detectors. Resolve those names on first access instead.
from .register_dense_matcher import get_dense_matcher
from .specs import (
    PointDenseMatchingOptions,
    LineDenseMatchingOptions,
    GroupDenseMatchingOptions,
    DenseMatchingOptions,
)

_LAZY_ATTRS = {
    "DenseMatchingResult": "base_dense_matcher",
    "BiDenseMatchingResult": "base_dense_matcher",
    "compute_point_distance_matrix": "metrics",
    "compute_line_distance_matrix": "metrics",
    "compute_mask_overlap_matrix": "metrics",
    "match_points_via_dense_matching": "associate",
    "match_lines_via_dense_matching": "associate",
    "match_groups_via_dense_matching": "associate",
    "associate_via_dense_matching": "associate",
}

__all__ = [
    "DenseMatchingResult",
    "BiDenseMatchingResult",
    "compute_point_distance_matrix",
    "compute_line_distance_matrix",
    "compute_mask_overlap_matrix",
    "get_dense_matcher",
    "PointDenseMatchingOptions",
    "LineDenseMatchingOptions",
    "GroupDenseMatchingOptions",
    "DenseMatchingOptions",
    "match_points_via_dense_matching",
    "match_lines_via_dense_matching",
    "match_groups_via_dense_matching",
    "associate_via_dense_matching",
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
