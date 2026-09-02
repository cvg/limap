import numpy as np
from typeguard import typechecked
from pathlib import Path

import limap.util.io as limapio


@typechecked
def get_group_mask_filename(folder: Path, image_id: int) -> Path:
    return folder / f"mask_{image_id}.npy"


@typechecked
def read_group_mask(folder: Path, image_id: int) -> np.ndarray:
    path = get_group_mask_filename(folder, image_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Mask for image_id={image_id} not found at {path}"
        )
    return limapio.read_npy(path)


@typechecked
def read_group_masks(folder: Path) -> dict[int, np.ndarray]:
    masks = {}
    for file in folder.glob("mask_*.npy"):
        stem = file.stem  # e.g. "mask_12"
        try:
            img_id = int(stem[len("mask_") :])
        except ValueError:
            continue
        masks[img_id] = limapio.read_npy(file)
    return masks


@typechecked
def write_group_mask(folder: Path, image_id: int, mask: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mask_{image_id}.npy"
    limapio.save_npy(path, mask)


@typechecked
def write_group_masks(folder: Path, masks: dict[int, np.ndarray]):
    folder.mkdir(parents=True, exist_ok=True)
    for img_id, mask in masks.items():
        write_group_mask(folder, img_id, mask)
