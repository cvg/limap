import pytlsd

from ..base_detector import (
    BaseDetector,
    DefaultDetectorOptions,
)


class LSDDetector(BaseDetector):
    def __init__(self, options=DefaultDetectorOptions):
        super().__init__(options)

    def get_module_name(self):
        return "lsd"

    def detect(self, camview):
        img = camview.read_image(set_gray=self.set_gray)
        segs = pytlsd.lsd(img)
        return segs
