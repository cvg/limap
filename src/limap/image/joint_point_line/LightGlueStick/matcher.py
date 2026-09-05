from ...line.base_matcher import DefaultMatcherOptions
from ..wireframe_matcher import WireframeLineMatcher
from .common import load_lightgluestick, run_lightgluestick


class LightGlueStickMatcher(WireframeLineMatcher):
    """The line half of LightGlueStick.

    Use
    :class:`~limap.image.joint_point_line.LightGlueStick.LightGlueStickJointMatcher`
    instead to keep the point matches the same forward pass produces.
    """

    def __init__(
        self,
        extractor,
        options=DefaultMatcherOptions,
        lightgluestick_options=None,
        device=None,
    ):
        super().__init__(extractor, options, device)
        self.net = load_lightgluestick(
            self.weight_path, self.device, lightgluestick_options
        )

    def get_module_name(self):
        return "lightgluestick"

    def _match(self, descinfo1, descinfo2):
        return run_lightgluestick(self.net, descinfo1, descinfo2, self.device)
