from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from ..line.register_detector import DetectorOptions, ExtractorOptions
from ..line.specs import LineDetectionOptions
from .register_joint_matcher import JointMatcherOptions


@dataclass
class JointPointLineDetectionOptions:
    """Options for detecting points and lines in a single network pass.

    Mirrors :class:`~limap.image.line.LineDetectionOptions`, minus the parts
    that do not apply: the joint detector is its own descriptor extractor, so
    there is no separate extractor method and descriptors are always computed.
    """

    skip_exists: bool = True
    weight_path: Path | None = None

    method: str = "upal"

    _detector_options: DetectorOptions = field(default_factory=DetectorOptions)

    @property
    def detector_options(self) -> DetectorOptions:
        options = deepcopy(self._detector_options)
        if self.weight_path is not None:
            options.base_options.weight_path = self.weight_path
        return options

    @detector_options.setter
    def detector_options(self, opts: DetectorOptions) -> None:
        self._detector_options = deepcopy(opts)

    def as_line_detection_options(self) -> LineDetectionOptions:
        """The line-side view of this joint detector.

        Association builds its line matcher's extractor from
        :class:`LineDetectionOptions`. A joint detector is both the detector
        and the extractor, so both methods are its own.
        """
        options = LineDetectionOptions(
            skip_exists=self.skip_exists,
            weight_path=self.weight_path,
            detector_method=self.method,
            extractor_method=self.method,
        )
        options.detector_options = self._detector_options
        options.extractor_options = ExtractorOptions(
            base_options=deepcopy(self._detector_options.base_options),
            upal_options=deepcopy(self._detector_options.upal_options),
        )
        return options


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
