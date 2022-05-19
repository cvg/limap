
def get_detector(cfg_detector, max_num_2d_segs=3000):
    method = cfg_detector["method"]
    if method == "lsd":
        from .LSD import LSDDetector
        return LSDDetector(set_gray=True, max_num_2d_segs=max_num_2d_segs, n_jobs=cfg_detector["n_jobs"])
    elif method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(set_gray=True, max_num_2d_segs=max_num_2d_segs, n_jobs=cfg_detector["n_jobs"])
    else:
        raise NotImplementedError

def get_extractor(cfg_extractor):
    method = cfg_extractor["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Detector
        return SOLD2Detector(set_gray=True, n_jobs=cfg_extractor["n_jobs"])
    else:
        raise NotImplementedError

