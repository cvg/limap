# register_joint_matcher and specs are cheap; base_joint_matcher and process
# reach for h5py, hloc and torch, which are optional dependencies arriving with
# the git-sourced detectors. Resolve those names on first access instead.
from .register_joint_matcher import (
    JointMatcherOptions,
    get_joint_matcher,
    get_joint_matcher_class,
)
from .specs import (
    JointPointLineDetectionOptions,
    JointPointLineMatcherOptions,
)

_LAZY_ATTRS = {
    "BaseJointMatcherOptions": "base_joint_matcher",
    "BaseJointMatcher": "base_joint_matcher",
    "JointMatchResult": "base_joint_matcher",
    "remap_point_matches": "base_joint_matcher",
    "joint_point_line_detection": "process",
    "joint_point_line_description": "process",
    "joint_point_line_matching": "process",
    "write_hloc_features": "base_joint_matcher",
}

__all__ = [
    "BaseJointMatcher",
    "BaseJointMatcherOptions",
    "JointMatchResult",
    "JointMatcherOptions",
    "JointPointLineDetectionOptions",
    "JointPointLineMatcherOptions",
    "get_joint_matcher",
    "get_joint_matcher_class",
    "joint_point_line_description",
    "joint_point_line_detection",
    "joint_point_line_matching",
    "remap_point_matches",
    "write_hloc_features",
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
