from dataclasses import dataclass, field
from pathlib import Path

from limap._limap._image._groups import GroupVotingOptions

from .point import PointDetectionOptions, PointMatcherOptions
from .line import LineDetectionOptions, LineMatcherOptions
from .groups import GroupDescriptionOptions
from .dense_matcher import DenseMatchingOptions
from .joint_point_line import JointPointLineMatcherOptions


@dataclass
class ImageDescriptionOptions:
    point_detection: PointDetectionOptions = field(
        default_factory=PointDetectionOptions
    )
    line_detection: LineDetectionOptions = field(
        default_factory=LineDetectionOptions
    )
    use_joint_point_line_detection: bool = False

    # Set when a joint point-line matcher will run: the point features are
    # then taken from the line description rather than detected separately,
    # since the matcher's junctions are that same pass's keypoints. Derived by
    # the runners from ImageAssociationOptions, not meant to be set in a config.
    joint_point_line_matcher: JointPointLineMatcherOptions | None = None

    # Skip point detection entirely (use existing points from reconstruction)
    skip_point_detection: bool = False

    skip_group_description: bool = False
    group_description: GroupDescriptionOptions = field(
        default_factory=GroupDescriptionOptions
    )


@dataclass
class ImageAssociationOptions:
    # Skip point matching entirely (use existing points from reconstruction)
    skip_point_matching: bool = False

    # Whether to perform geometric verification on keypoints.
    # Now only enabled when using classical feature matching
    geometric_verification: bool = True

    # Point geometric verification mode:
    #   None  → auto-detect (pose-guided if posed, E/F if unposed)
    #   True  → force pose-guided verification
    #   False → force E/F-based verification (RANSAC)
    pose_guided_point_verification: bool | None = None

    # use dense matching
    use_dense_matching: bool = False
    use_dense_matching_for_groups: bool = False
    dense_matching_options: DenseMatchingOptions = field(
        default_factory=DenseMatchingOptions
    )

    # use classical feature matching
    # Match points and lines in a single pass, replacing both point_matcher
    # and line_matcher.
    use_joint_point_line_matcher: bool = False
    joint_point_line_matcher: JointPointLineMatcherOptions = field(
        default_factory=JointPointLineMatcherOptions
    )
    point_descriptor_path: Path | None = None
    point_matcher: PointMatcherOptions = field(
        default_factory=PointMatcherOptions
    )
    line_descriptor_path: Path | None = None
    line_matcher: LineMatcherOptions = field(default_factory=LineMatcherOptions)
    skip_line_matching: bool = False
    skip_group_matching: bool = False

    # Voting for unmatched groups
    skip_group_voting: bool = False
    group_voting: GroupVotingOptions = field(default_factory=GroupVotingOptions)

    # VP geometric verification (requires poses)
    vp_geometric_verification: bool = True
    vp_verification_threshold: float = 10.0  # degrees

    # Plane geometric verification (requires poses)
    plane_geometric_verification: bool = True
    plane_verification_threshold: float = 30.0  # degrees
