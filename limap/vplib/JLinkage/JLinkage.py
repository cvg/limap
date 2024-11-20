from _limap import _vplib

from ..base_vp_detector import BaseVPDetector, DefaultVPDetectorOptions


class JLinkage(BaseVPDetector):
    def __init__(self, cfg_jlinkage, options=DefaultVPDetectorOptions):
        super().__init__(options)
        self.detector = _vplib.JLinkage(cfg_jlinkage)

    def get_module_name(self):
        return "JLinkage"

    def detect_vp(self, lines, camview=None):
        vpresult = self.detector.AssociateVPs(lines)
        return vpresult

    # comments for unification of the interfaces with n_jobs
    # # parallelization directly in cpp is faster at initializing threads
    # def detect_vp_all_images(self, all_lines, camviews=None):
    #     return self.detector.AssociateVPsParallel(all_lines)
