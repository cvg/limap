from ..base_joint_matcher import DefaultJointMatcherOptions
from ..wireframe_matcher import WireframeJointMatcher
from .common import load_lightgluestick, run_lightgluestick


class LightGlueStickJointMatcher(WireframeJointMatcher):
    """LightGlueStick, keeping the point matches of the same forward pass.

    Matches the same junctions as
    :class:`~limap.image.joint_point_line.GlueStick.GlueStickJointMatcher`,
    with a lighter network, so the two are interchangeable over the same
    wireframe descriptors.
    """

    def __init__(
        self,
        options=DefaultJointMatcherOptions,
        lightgluestick_options=None,
        device=None,
    ):
        super().__init__(options, device)
        self.net = load_lightgluestick(
            self.weight_path, self.device, lightgluestick_options
        )

    def get_module_name(self):
        return "lightgluestick"

    def _match(self, descinfo1, descinfo2):
        return run_lightgluestick(self.net, descinfo1, descinfo2, self.device)
