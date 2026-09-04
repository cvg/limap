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
    "upal": 2.0,
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
class UPALOptions:
    """Options for the UPAL joint point-line network.

    The keypoints double as the seeds for point-seeded LSD. The number of
    segments is capped by ``BaseDetectorOptions.max_num_2d_segs`` like every
    other detector; these are upstream's own filtering defaults.
    """

    # UPAL's detector is top-k with no score threshold, so this is a hard
    # count, not a cap: every image returns exactly this many keypoints. For
    # scale, aliked-n16 is threshold-based and yields a few hundred.
    max_num_keypoints: int = 4096
    min_line_length: float = 25.0
    max_mean_distance: float = 2.0


@dataclass
class DetectorOptions:
    base_options: BaseDetectorOptions = field(
        default_factory=BaseDetectorOptions
    )
    upal_options: UPALOptions = field(default_factory=UPALOptions)


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
    elif method == "upal":
        from ..joint_point_line.UPAL import UPALDetector

        return UPALDetector(options, loptions.upal_options)
    else:
        raise NotImplementedError


@dataclass
class ExtractorOptions:
    base_options: BaseDetectorOptions = field(
        default_factory=BaseDetectorOptions
    )
    upal_options: UPALOptions = field(default_factory=UPALOptions)


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
        from .GlueStick import WireframeExtractor

        return WireframeExtractor(options)
    elif method == "dense_naive":
        from .dense import DenseNaiveExtractor

        return DenseNaiveExtractor(options)
    elif method == "upal":
        from ..joint_point_line.UPAL import UPALDetector

        return UPALDetector(options, loptions.upal_options)
    else:
        raise NotImplementedError
