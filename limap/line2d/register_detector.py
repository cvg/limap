from .base_detector import BaseDetectorOptions

def get_detector(cfg_detector, max_num_2d_segs=3000, do_merge_lines=False):
    options = BaseDetectorOptions()
    options = options._replace(set_gray = True, max_num_2d_segs = max_num_2d_segs, do_merge_lines = do_merge_lines)

    method = cfg_detector["method"]
    if method == "lsd":
        from .LSD import LSDDetector
        return LSDDetector(options)
    elif method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(options)
    else:
        raise NotImplementedError

def get_extractor(cfg_extractor):
    options = BaseDetectorOptions()
    options = options._replace(set_gray = True)

    method = cfg_extractor["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(options)
    elif method == "superglue_endpoints":
        from .superglue_endpoints import EndpointsExtractor
        return EndpointsExtractor(options)
    else:
        raise NotImplementedError

