from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy

from .register_detector import DetectorOptions, ExtractorOptions
from .register_matcher import MatcherOptions


@dataclass
class LineDetectionOptions:
    skip_exists: bool = True
    compute_descinfo: bool = True
    weight_path: Path | None = None

    detector_method: str = "deeplsd"
    extractor_method: str | None = "wireframe"

    _detector_options: DetectorOptions = field(default_factory=DetectorOptions)
    _extractor_options: ExtractorOptions = field(
        default_factory=ExtractorOptions
    )

    @property
    def detector_options(self) -> DetectorOptions:
        options = deepcopy(self._detector_options)
        if self.weight_path is not None:
            options.base_options.weight_path = self.weight_path
        return options

    @detector_options.setter
    def detector_options(self, opts: DetectorOptions) -> None:
        self._detector_options = deepcopy(opts)

    @property
    def extractor_options(self) -> ExtractorOptions:
        options = deepcopy(self._extractor_options)
        if self.weight_path is not None:
            options.base_options.weight_path = self.weight_path
        return options

    @extractor_options.setter
    def extractor_options(self, opts: ExtractorOptions) -> None:
        self._extractor_options = deepcopy(opts)


@dataclass
class LineMatcherOptions:
    skip_exists: bool = True
    weight_path: Path | None = None

    method: str = "gluestick"
    _matching_options: MatcherOptions = field(default_factory=MatcherOptions)

    @property
    def matching_options(self) -> MatcherOptions:
        options = deepcopy(self._matching_options)
        if self.weight_path is not None:
            options.base_options.weight_path = self.weight_path
        return options

    @matching_options.setter
    def matching_options(self, opts: MatcherOptions) -> None:
        self._matching_options = deepcopy(opts)
