from dataclasses import dataclass, field


@dataclass
class PointDenseMatchingOptions:
    pixel_thresh: float = 4.0


@dataclass
class LineDenseMatchingOptions:
    pixel_thresh: float = 4.0
    n_samples: float = 21
    min_segment_overlap: float = 0.2


@dataclass
class GroupDenseMatchingOptions:
    overlap_thresh: float = 0.6
    min_num_pixels: float = 1000


@dataclass
class DenseMatchingOptions:
    method: str = "tiny_roma"
    point_matching: PointDenseMatchingOptions = field(
        default_factory=PointDenseMatchingOptions
    )
    line_matching: LineDenseMatchingOptions = field(
        default_factory=LineDenseMatchingOptions
    )
    group_matching: GroupDenseMatchingOptions = field(
        default_factory=GroupDenseMatchingOptions
    )
    skip_point_matching: bool = False
    skip_line_matching: bool = False
    skip_group_matching: bool = False
    save_warps: bool = False
