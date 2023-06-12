import cv2

class BaseDepthReader():
    def __init__(self, filename):
        self.filename = filename

    def read(self, filename):
        raise NotImplementedError

    def read_depth(self, img_hw=None):
        depth = self.read(self.filename)
        if img_hw is not None and (depth.shape[0] != img_hw[0] or depth.shape[1] != img_hw[1]):
            depth = cv2.resize(depth, (img_hw[1], img_hw[0]))
        return depth

