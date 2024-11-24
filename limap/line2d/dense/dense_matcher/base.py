import torch


class BaseDenseMatcher:
    def __init__(self):
        pass

    def to_normalized_coordinates(self, coords, h, w):
        """
        coords: (..., 2) in the order x, y
        """
        coords_x = 2 / w * coords[..., 0] - 1
        coords_y = 2 / h * coords[..., 1] - 1
        return torch.stack([coords_x, coords_y], axis=-1)

    def to_unnormalized_coordinates(self, coords, h, w):
        """
        Inverse operation of `to_normalized_coordinates`
        """
        coords_x = (coords[..., 0] + 1) * w / 2
        coords_y = (coords[..., 1] + 1) * h / 2
        return torch.stack([coords_x, coords_y], axis=-1)

    def get_sample_thresh(self):
        """
        return sample threshold
        """
        raise NotImplementedError

    def get_warping_symmetric(self, img1, img2):
        """
        return warp_1to2 ([-1, 1]), cert_1to2, warp_2to1([-1, 1]), cert_2to1
        """
        raise NotImplementedError
