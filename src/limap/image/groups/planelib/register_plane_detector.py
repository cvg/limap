from dataclasses import dataclass, field

from .base_plane_detector import BasePlaneDetectorOptions


@dataclass
class PxwPlanarOptions:
    """
    Options for the MoGe 4-head planarity detector. The segmentation
    defaults are the canonical parameters of the pxwplanar benchmark.
    """

    # local .pt checkpoint or a Hugging Face repo id
    model_path: str = "alpayozkan/pxwplanar-moge2-planarity"
    device: str = "cuda"
    num_tokens: int = 1600
    threshold_planarity: float = 0.3
    normal_threshold_deg: float = 5.0
    depth_threshold: float = 0.025  # relative to the center depth
    neighbor_match_count_thresh: int = 8


@dataclass
class DetectorOptions:
    base_options: BasePlaneDetectorOptions = field(
        default_factory=BasePlaneDetectorOptions
    )
    pxwplanar_options: PxwPlanarOptions = field(
        default_factory=PxwPlanarOptions
    )


def get_plane_detector(method: str, plane_options: DetectorOptions):
    """
    Get a plane detector
    """
    options = plane_options.base_options
    if method == "pxwplanar":
        from .pxwplanar import PxwPlanar

        return PxwPlanar(options, plane_options.pxwplanar_options)
    else:
        raise NotImplementedError
