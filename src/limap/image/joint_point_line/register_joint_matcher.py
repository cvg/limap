from dataclasses import dataclass, field

from .base_joint_matcher import BaseJointMatcherOptions


@dataclass
class JointMatcherOptions:
    base_options: BaseJointMatcherOptions = field(
        default_factory=BaseJointMatcherOptions
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
    # TODO: support LightGlueStick, which matches the same junctions with a
    # lighter network and needs no change outside this branch.
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
    return get_joint_matcher_class(method)(loptions.base_options)
