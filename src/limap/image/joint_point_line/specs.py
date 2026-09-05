from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from .register_joint_matcher import JointMatcherOptions


@dataclass
class JointPointLineMatcherOptions:
    """Options for matching points and lines in a single network pass.

    Mirrors :class:`~limap.image.line.LineMatcherOptions`. The matcher
    replaces both the point matcher and the line matcher, so
    ``point_matcher`` and ``line_matcher`` are unused when it runs.
    """

    skip_exists: bool = True
    weight_path: Path | None = None

    method: str = "gluestick"

    _matching_options: JointMatcherOptions = field(
        default_factory=JointMatcherOptions
    )

    @property
    def matching_options(self) -> JointMatcherOptions:
        options = deepcopy(self._matching_options)
        if self.weight_path is not None:
            options.base_options.weight_path = self.weight_path
        return options

    @matching_options.setter
    def matching_options(self, opts: JointMatcherOptions) -> None:
        self._matching_options = deepcopy(opts)
