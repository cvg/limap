from .base_detector import BaseDetectorOptions
from dataclasses import dataclass, field


# Default 2D uncertainty values for each detector (in pixels).
# Represents the expected localization uncertainty of line endpoints.
_UNCERTAINTY2D_DEFAULTS: dict[str, float] = {
    "lsd": 2.0,
    "deeplsd": 4.0,
    "sold2": 5.0,
    "hawpv3": 5.0,
    "tp_lsd": 5.0,
}


def get_uncertainty2d(method: str) -> float:
    """Get the default 2D uncertainty for a line detector.

    Args:
        method: Name of the line detector (e.g., "lsd", "deeplsd", "sold2").

    Returns:
        Default 2D uncertainty value in pixels for the specified detector.

    Raises:
        ValueError: If the detector method is not recognized.
    """
    if method not in _UNCERTAINTY2D_DEFAULTS:
        raise ValueError(
            f"Unknown detector method: {method}. "
            f"Known methods: {list(_UNCERTAINTY2D_DEFAULTS.keys())}"
        )
    return _UNCERTAINTY2D_DEFAULTS[method]


@dataclass
class DetectorOptions:
    base_options: BaseDetectorOptions = field(
        default_factory=BaseDetectorOptions
    )


def get_detector(method: str, loptions: DetectorOptions):
    options = loptions.base_options
    if method == "lsd":
        from .LSD import LSDDetector

        return LSDDetector(options)
    elif method == "sold2":
        from .SOLD2 import SOLD2Detector

        return SOLD2Detector(options)
    elif method == "hawpv3":
        from .HAWPv3 import HAWPv3Detector

        return HAWPv3Detector(options)
    elif method == "tp_lsd":
        from .TP_LSD import TPLSDDetector

        return TPLSDDetector(options)
    elif method == "deeplsd":
        from .DeepLSD import DeepLSDDetector

        return DeepLSDDetector(options)
    else:
        raise NotImplementedError


@dataclass
class ExtractorOptions:
    base_options: BaseDetectorOptions = field(
        default_factory=BaseDetectorOptions
    )


def get_extractor(method: str, loptions: ExtractorOptions):
    """
    Get a line descriptor speicified by cfg_extractor["method"]

    Args:
        cfg_extractor: config for the line extractor
    """
    options = loptions.base_options
    if method == "sold2":
        from .SOLD2 import SOLD2Detector

        return SOLD2Detector(options)
    elif method == "lbd":
        from .LBD import LBDExtractor

        return LBDExtractor(options)
    elif method == "linetr":
        from .LineTR import LineTRExtractor

        return LineTRExtractor(options)
    elif method == "l2d2":
        from .L2D2 import L2D2Extractor

        return L2D2Extractor(options)
    elif method == "superpoint_endpoints":
        from .endpoints import SuperPointEndpointsExtractor

        return SuperPointEndpointsExtractor(options)
    elif method == "wireframe":
        from ..joint_point_line.GlueStick import WireframeExtractor

        return WireframeExtractor(options)
    elif method == "dense_naive":
        from .dense import DenseNaiveExtractor

        return DenseNaiveExtractor(options)
    else:
        raise NotImplementedError
