class BaseP3DReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self, filename):
        """
        Virtual method - Read a point cloud from a filename

        Args:
            filename (str): The filename of the depth image
        Returns:
            point cloud (:class:`np.array` of shape (N, 3)): \
                The array for the 3D points
        """
        raise NotImplementedError

    def read_p3ds(self):
        """
        Read a point cloud using the read(self, filename) function

        Returns:
            point cloud (:class:`np.array` of shape (N, 3)): \
                The array for the 3D points
        """
        p3ds = self.read(self.filename)
        return p3ds
