from .base_matcher import BaseMatcherOptions

def get_matcher(cfg_matcher, extractor, n_neighbors=20):
    options = BaseMatcherOptions()
    options = options._replace(n_neighbors = n_neighbors, topk = cfg_matcher["topk"], n_jobs = cfg_matcher["n_jobs"])

    method = cfg_matcher["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Matcher
        return SOLD2Matcher(extractor, options)
    elif method == "nn_endpoints":
        from .endpoints import NNEndpointsMatcher
        return NNEndpointsMatcher(extractor, options)
    elif method == "superglue_endpoints":
        from .endpoints import SuperGlueEndpointsMatcher
        return SuperGlueEndpointsMatcher(extractor, options, weights=cfg_matcher["superglue"]["weights"])
    else:
        raise NotImplementedError
