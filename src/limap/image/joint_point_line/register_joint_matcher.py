from dataclasses import dataclass, field

from ..line.register_matcher import LightGlueStickOptions
from .base_joint_matcher import BaseJointMatcherOptions


@dataclass
class JointMatcherOptions:
    base_options: BaseJointMatcherOptions = field(
        default_factory=BaseJointMatcherOptions
    )
    lightgluestick_options: LightGlueStickOptions = field(
        default_factory=LightGlueStickOptions
    )


def get_joint_matcher_class(method: str):
    """
    Get a joint point-line matcher class by method name

    The class rather than an instance, because the description step calls
    ``export_point_features`` on it before any weights are needed.

    Args:
        method (str): Name of the joint matcher
    Returns:
        A subclass of \
        :class:`~limap.image.joint_point_line.base_joint_matcher.BaseJointMatcher`
    """
    if method == "gluestick":
        from .GlueStick import GlueStickJointMatcher

        return GlueStickJointMatcher
    elif method == "lightgluestick":
        from .LightGlueStick import LightGlueStickJointMatcher

        return LightGlueStickJointMatcher
    else:
        raise NotImplementedError


def get_joint_matcher(method: str, loptions: JointMatcherOptions):
    """
    Get a joint point-line matcher by method name

    Args:
        method (str): Name of the joint matcher
        loptions (:class:`JointMatcherOptions`): Options
    Returns:
        The joint matcher, a subclass of \
        :class:`~limap.image.joint_point_line.base_joint_matcher.BaseJointMatcher`
    """
    matcher_class = get_joint_matcher_class(method)
    if method == "lightgluestick":
        return matcher_class(
            loptions.base_options, loptions.lightgluestick_options
        )
    return matcher_class(loptions.base_options)
