
def get_matcher(cfg_matcher, extractor, n_neighbors=20):
    method = cfg_matcher["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Matcher
        return SOLD2Matcher(extractor, n_neighbors=n_neighbors, topk=cfg_matcher["topk"], n_jobs=cfg_matcher["n_jobs"])
    elif method == "superglue_endpoints":
        from .superglue_endpoints import SuperGlueEndpointsMatcher
        return SuperGlueEndpointsMatcher(extractor, weights=cfg_matcher["weights"], n_neighbors=n_neighbors, topk=cfg_matcher["topk"], n_jobs=cfg_matcher["n_jobs"])
    else:
        raise NotImplementedError
