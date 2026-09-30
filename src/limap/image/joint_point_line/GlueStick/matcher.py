from ...line.base_matcher import DefaultMatcherOptions
from ..wireframe_matcher import WireframeLineMatcher
from .common import load_gluestick, run_gluestick


class GlueStickMatcher(WireframeLineMatcher):
    """The line half of GlueStick.

    Use :class:`~limap.image.joint_point_line.GlueStick.GlueStickJointMatcher`
    instead to keep the point matches the same forward pass produces.
    """

    def __init__(self, extractor, options=DefaultMatcherOptions, device=None):
        super().__init__(extractor, options, device)
        self.net = load_gluestick(self.weight_path, self.device)

    def get_module_name(self):
        return "gluestick"

    def _match(self, descinfo1, descinfo2):
        return run_gluestick(self.net, descinfo1, descinfo2, self.device)
