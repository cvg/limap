from limap._limap._image._groups import _vplib
from limap._limap._image._groups._vplib import BaseVPDetectorOptions

from dataclasses import dataclass, field


@dataclass
class JLinkageOptions:
    th_perp_supports: float = 3.0


@dataclass
class DetectorOptions:
    base_options: BaseVPDetectorOptions = field(
        default_factory=BaseVPDetectorOptions
    )
    jlinkage_options: JLinkageOptions = field(default_factory=JLinkageOptions)


def get_vp_detector(method: str, vpoptions: DetectorOptions):
    """
    Get a vanishing point detector
    """
    options = vpoptions.base_options
    if method == "jlinkage":
        from .JLinkage import JLinkage

        return JLinkage(
            _vplib.JLinkageOptions(
                base_options=options,
                th_perp_supports=vpoptions.jlinkage_options.th_perp_supports,
            )
        )
    elif method == "progressivex":
        from .progressivex import ProgressiveX

        return ProgressiveX(options)
    else:
        raise NotImplementedError
