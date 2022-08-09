
def get_detector(cfg_detector, max_num_2d_segs=3000):
    method = cfg_detector["method"]
    if method == "lsd":
        from .LSD import LSDDetector
        return LSDDetector(set_gray=True, max_num_2d_segs=max_num_2d_segs)
    elif method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(set_gray=True, max_num_2d_segs=max_num_2d_segs)
    else:
        raise NotImplementedError

def get_extractor(cfg_extractor):
    method = cfg_extractor["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(set_gray=True)
    elif method == "superglue_endpoints":
        from .superglue_endpoints import EndpointsExtractor
        return EndpointsExtractor(set_gray=True)
    else:
        raise NotImplementedError

