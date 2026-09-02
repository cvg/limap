from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path

from . import vplib
from . import planelib


@dataclass
class VPDetectionOptions:
    skip_exists: bool = True
    method: str = "jlinkage"
    options: vplib.DetectorOptions = field(
        default_factory=vplib.DetectorOptions
    )


@dataclass
class PlaneDetectionOptions:
    weight_path: Path = Path.home() / ".limap" / "models"
    skip_exists: bool = True
    method: str = "pxwplanar"
    visualize: bool = True
    _options: planelib.DetectorOptions = field(
        default_factory=planelib.DetectorOptions
    )

    @property
    def options(self) -> planelib.DetectorOptions:
        options = deepcopy(self._options)
        options.base_options.weight_path = self.weight_path
        return options

    @options.setter
    def options(self, opts: planelib.DetectorOptions) -> None:
        self._options = deepcopy(opts)


@dataclass
class GroupDescriptionOptions:
    detect_vp: bool = True
    vp_detection: VPDetectionOptions = field(default_factory=VPDetectionOptions)

    detect_plane: bool = True
    plane_detection: PlaneDetectionOptions = field(
        default_factory=PlaneDetectionOptions
    )
    plane_min_line_overlap_length: float = 40
    plane_dilation_radius: float = 2
