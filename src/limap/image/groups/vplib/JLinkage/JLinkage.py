from limap._limap._image._groups import _vplib

from ..base_vp_detector import BaseVPDetector

DefaultJLinkageOptions = _vplib.JLinkageOptions()


class JLinkage(BaseVPDetector):
    def __init__(self, options=DefaultJLinkageOptions):
        super().__init__(options.base_options)
        self.detector = _vplib.JLinkage(options)

    def get_module_name(self):
        return "JLinkage"

    def detect_vp(self, lines, image_path=None):
        return self.detector.associate_vps(lines)

    # parallelization directly in cpp is faster at initializing threads
    def detect_vp_all_images(self, all_lines, image_paths=None):
        return self.detector.associate_vps_parallel(all_lines)
