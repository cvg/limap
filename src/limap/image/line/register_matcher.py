from .base_matcher import BaseMatcherOptions
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SuperGlueMatcherOptions:
    weights: str = "outdoor"


@dataclass
class DenseMatcherOptions:
    one_to_many: bool = False
    weights: str = "outdoor"


@dataclass
class MatcherOptions:
    base_options: BaseMatcherOptions = field(default_factory=BaseMatcherOptions)
    superglue_options: SuperGlueMatcherOptions = field(
        default_factory=SuperGlueMatcherOptions
    )
    dense_options: DenseMatcherOptions = field(
        default_factory=DenseMatcherOptions
    )


def get_matcher(method: str, loptions: MatcherOptions, extractor: Any):
    options = loptions.base_options
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
            extractor, options, weights=loptions.superglue_options.weights
        )
    elif method == "upal":
        from ..joint_point_line.UPAL import UPALMatcher

        return UPALMatcher(extractor, options)
    elif method == "gluestick":
        from .GlueStick import GlueStickMatcher

        return GlueStickMatcher(extractor, options)
    elif method == "dense_roma":
        from .dense import BaseDenseLineMatcherOptions, RoMaLineMatcher

        dense_options = BaseDenseLineMatcherOptions()
        dense_options = dense_options._replace(
            one_to_many=loptions.dense_options.one_to_many
        )
        return RoMaLineMatcher(
            extractor,
            options=options,
            dense_options=dense_options,
            mode=loptions.dense_options.weights,
        )
    else:
        raise NotImplementedError
