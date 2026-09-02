from dataclasses import dataclass
import torch


@dataclass
class DenseMatchingResult:
    device: str = "cpu"
    sample_threshold: float | None = None
    source_shape: tuple[int, int] | None = None  # h, w
    target_shape: tuple[int, int] | None = None  # h, w

    # with value range [-1, 1]
    warp: torch.Tensor | None = None

    # uncertainty of warp_1to2
    certainty: torch.Tensor | None = None

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

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "sample_threshold": self.sample_threshold,
            "source_shape": self.source_shape,
            "target_shape": self.target_shape,
            "warp": self.warp.cpu().numpy() if self.warp is not None else None,
            "certainty": self.certainty.cpu().numpy()
            if self.certainty is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DenseMatchingResult":
        device = data.get("device", "cpu")
        warp = (
            torch.from_numpy(data["warp"]).to(device)
            if data.get("warp") is not None
            else None
        )
        certainty = (
            torch.from_numpy(data["certainty"]).to(device)
            if data.get("certainty") is not None
            else None
        )
        return cls(
            device=device,
            sample_threshold=data.get("sample_threshold"),
            source_shape=data.get("source_shape"),
            target_shape=data.get("target_shape"),
            warp=warp,
            certainty=certainty,
        )


@dataclass
class BiDenseMatchingResult:
    match_1to2: DenseMatchingResult
    match_2to1: DenseMatchingResult

    def to_dict(self) -> dict:
        return {
            "match_1to2": self.match_1to2.to_dict(),
            "match_2to1": self.match_2to1.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BiDenseMatchingResult":
        return cls(
            match_1to2=DenseMatchingResult.from_dict(data["match_1to2"]),
            match_2to1=DenseMatchingResult.from_dict(data["match_2to1"]),
        )


class BaseDenseMatcher:
    def __init__(self):
        pass

    def get_warping(self, img1, img2) -> DenseMatchingResult:
        """
        return DenseMatchingResult from img1 to img2
        """
        raise NotImplementedError

    def get_warping_symmetric(self, img1, img2) -> BiDenseMatchingResult:
        """
        return BiDenseMatchingResult
        """
        raise NotImplementedError
