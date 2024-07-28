from .base_detector import BaseDetectorOptions


def get_detector(
    cfg_detector,
    max_num_2d_segs=3000,
    do_merge_lines=False,
    visualize=False,
    weight_path=None,
):
    """
    Get a line detector specified by cfg_detector["method"]

    Args:
        cfg_detector: config for the line detector
    """
    options = BaseDetectorOptions()
    options = options._replace(
        set_gray=True,
        max_num_2d_segs=max_num_2d_segs,
        do_merge_lines=do_merge_lines,
        visualize=visualize,
        weight_path=weight_path,
    )

    method = cfg_detector["method"]
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


def get_extractor(cfg_extractor, weight_path=None):
    """
    Get a line descriptor speicified by cfg_extractor["method"]

    Args:
        cfg_extractor: config for the line extractor
    """
    options = BaseDetectorOptions()
    options = options._replace(set_gray=True, weight_path=weight_path)

    method = cfg_extractor["method"]
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
    else:
        raise NotImplementedError
