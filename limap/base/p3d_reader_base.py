class BaseP3DReader():
    def __init__(self, filename):
        self.filename = filename

    def read(self, filename):
        raise NotImplementedError

    def read_p3ds(self):
        p3ds = self.read(self.filename)
        return p3ds

