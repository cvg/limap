from typing import Literal

from typeguard import typechecked

from limap._limap._scene import COLMAPMVSModel


@typechecked
def compute_neighbors(
    model: COLMAPMVSModel,
    n_neighbors: int,
    min_triangulation_angle: float = 1.0,
    neighbor_type: Literal["iou", "overlap", "dice"] = "iou",
) -> dict[int, list[int]]:
    """
    Compute neighboring images for each image based on point visibility.

    Args:
        model: The COLMAP MVS model containing images and points.
        n_neighbors: Number of neighbors to compute for each image.
        min_triangulation_angle: Minimum triangulation angle in degrees.
        neighbor_type: Type of neighbor selection metric:
            - "iou": Intersection over Union of visible points
            - "overlap": Number of shared visible points
            - "dice": Dice coefficient of visible points

    Returns:
        Dictionary mapping image ID to list of neighbor image IDs.
    """
    if neighbor_type == "iou":
        neighbors = model.get_max_iou_images(
            n_neighbors, min_triangulation_angle
        )
    elif neighbor_type == "overlap":
        neighbors = model.get_max_overlap_images(
            n_neighbors, min_triangulation_angle
        )
    elif neighbor_type == "dice":
        neighbors = model.get_max_dice_coeff_images(
            n_neighbors, min_triangulation_angle
        )
    else:
        raise NotImplementedError(f"Unknown neighbor_type: {neighbor_type}")
    return neighbors
