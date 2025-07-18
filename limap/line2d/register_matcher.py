from .base_matcher import BaseMatcherOptions


def get_matcher(cfg_matcher, extractor, n_neighbors=20, weight_path=None):
    """
    Get a line matcher specified by cfg_matcher["method"]

    Args:
        cfg_matcher: config for line matcher
        extractor: line extractor inherited from \
            :class:`limap.line2d.base_matcher.BaseMatcher`
    """
    options = BaseMatcherOptions()
    options = options._replace(
        n_neighbors=n_neighbors,
        topk=cfg_matcher["topk"],
        n_jobs=cfg_matcher["n_jobs"],
        weight_path=weight_path,
    )

    method = cfg_matcher["method"]
    if method == "sold2":
        from .SOLD2 import SOLD2Matcher

        return SOLD2Matcher(extractor, options)
    elif method == "lbd":
        from .LBD import LBDMatcher

        return LBDMatcher(extractor, options)
    elif method == "linetr":
        from .LineTR import LineTRMatcher

        return LineTRMatcher(extractor, options)
    elif method == "l2d2":
        from .L2D2 import L2D2Matcher

        return L2D2Matcher(extractor, options)
    elif method == "nn_endpoints":
        from .endpoints import NNEndpointsMatcher

        return NNEndpointsMatcher(extractor, options)
    elif method == "superglue_endpoints":
        from .endpoints import SuperGlueEndpointsMatcher

        return SuperGlueEndpointsMatcher(
            extractor, options, weights=cfg_matcher["superglue"]["weights"]
        )
    elif method == "gluestick":
        from .GlueStick import GlueStickMatcher

        return GlueStickMatcher(extractor, options)
    elif method == "dense_roma":
        from .dense import BaseDenseLineMatcherOptions, RoMaLineMatcher

        dense_options = BaseDenseLineMatcherOptions()
        if "one_to_many" in cfg_matcher:
            dense_options = dense_options._replace(
                one_to_many=cfg_matcher["dense"]["one_to_many"]
            )
        return RoMaLineMatcher(
            extractor,
            options=options,
            dense_options=dense_options,
            mode=cfg_matcher["dense"]["weights"],
        )
    else:
        raise NotImplementedError
