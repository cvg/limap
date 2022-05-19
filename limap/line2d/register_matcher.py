
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

def get_matcher(cfg_matcher, extractor, n_neighbors=20):
    method = cfg_matcher["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Matcher
        return SOLD2Matcher(extractor, n_neighbors=n_neighbors, topk=cfg_matcher["topk"], n_jobs=cfg_matcher["n_jobs"])
    else:
        raise NotImplementedError

