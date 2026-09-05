from ..base_joint_matcher import DefaultJointMatcherOptions
from ..wireframe_matcher import WireframeJointMatcher
from .common import load_gluestick, run_gluestick


class GlueStickJointMatcher(WireframeJointMatcher):
    """GlueStick, keeping the point matches of the same forward pass."""

    def __init__(self, options=DefaultJointMatcherOptions, device=None):
        super().__init__(options, device)
        self.net = load_gluestick(self.weight_path, self.device)

    def get_module_name(self):
        return "gluestick"

    def _match(self, descinfo1, descinfo2):
        return run_gluestick(self.net, descinfo1, descinfo2, self.device)
