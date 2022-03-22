import numpy as np
from sklearn.linear_model._base import LinearModel

class Line3D:
    def __init__(self, ptn=None, vec=None, rot_mat=None):
        self.ptn = ptn
        self.vec = vec
        self.rot_mat = rot_mat

class LinearModelEstimator(LinearModel):
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self._residues = None
        self.rank_ = None
        self.singular_ = None
        self.line = None

    def compute_residues(self, points):
        # vv is also a rotation matrix that can transform
        # Now the X-Axis of the points is the lambda through the line, and the
        # other components are the distance to the line in each of the axes
        rotated_residues = (points - self.line.ptn).dot(self.line.rot_mat.T)[:, 1:]
        residues = np.linalg.norm(rotated_residues, axis=1)
        return residues

    def compute_model(self, points):
        # Check that the data has the correct form
        assert len(points.shape) == 2
        self.line = Line3D()

        # Centers data to have mean zero along axis 0
        self.line.ptn = points.mean(axis=0)
        centered_data = points - self.line.ptn

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(centered_data)
        self.line.rot_mat = vv
        self.line.vec = vv[0]

    def fit(self, points, sample_weight=None):
        self.compute_model(points)
        return self

    def score(self, points):
        residues = self.compute_residues(points)
        # return np.sum(residues ** 2)
        return np.sum(np.abs(residues))

    def residues(self, points):
        residues = self.compute_residues(points)
        # return residues ** 2
        return np.abs(residues)

