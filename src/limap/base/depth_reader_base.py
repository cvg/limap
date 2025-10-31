import cv2


class BaseDepthReader:
    """
    Base class for the depth reader storing the filename and \
    potentially other information
    """

    def __init__(self, filename):
        self.filename = filename

    def read(self, filename):
        """
        Virtual method - Read depth from a filename

        Args:
            filename (str): The filename of the depth image
        Returns:
            depth (:class:`np.array` of shape (H, W)): \
                The array for the depth map
        """
        raise NotImplementedError

    def read_depth(self, img_hw=None):
        """
        Read depth using the read(self, filename) function

        Args:
            img_hw (pair of int, optional): \
                The height and width for the read depth. \
                By default we keep the original resolution of the file
        Returns:
            depth (:class:`np.array` of shape (H, W)): \
                The array for the depth map
        """
        depth = self.read(self.filename)
        if img_hw is not None and (
            depth.shape[0] != img_hw[0] or depth.shape[1] != img_hw[1]
        ):
            depth = cv2.resize(depth, (img_hw[1], img_hw[0]))
        return depth
