import cv2

import pycolmap
from pycolmap import logging
from dataclasses import dataclass, field
from pathlib import Path
from typeguard import typechecked

import limap.scene as scene
import limap.util.io as limapio
from limap.image.specs import PointDetectionOptions, PointMatcherOptions
from limap.util.types import Ranges

from .automatic_point_triangulation import AutomaticPointTriangulationOptions


@dataclass
class PairGenerationOptions:
    n_neighbors: int = 20
    min_triangulation_angle: float = 1.0
    neighbor_type: str = "dice"


@dataclass
class RangeCalculatorOptions:
    robust_range_min: float = 0.05
    robust_range_max: float = 0.95
    stretch_ratio: float = 1.25


@dataclass
class MetaInfoComputerOptions:
    cache_file: Path | None = None
    point_triangulation: AutomaticPointTriangulationOptions = field(
        default_factory=lambda: (
            MetaInfoComputerOptions._make_fast_point_triangulation_options()
        )
    )
    pair_generation: PairGenerationOptions = field(
        default_factory=PairGenerationOptions
    )
    range_calculator: RangeCalculatorOptions = field(
        default_factory=RangeCalculatorOptions
    )

    @staticmethod
    def _make_fast_point_triangulation_options() -> (
        AutomaticPointTriangulationOptions
    ):
        return AutomaticPointTriangulationOptions(
            point_detection=PointDetectionOptions(method="aliked-n16"),
            point_matcher=PointMatcherOptions(method="NN-mutual"),
        )


@typechecked
def check_valid_reconstruction(
    image_dir: Path, recon: pycolmap.Reconstruction
) -> bool:
    for img in recon.images.values():
        rel_path = Path(img.name)
        src = (image_dir / rel_path).resolve()
        if not src.exists():
            logging.error(
                f"Image not found: {src} "
                f"(image_dir={image_dir}, name={img.name})"
            )
            return False

        w, h = img.camera.width, img.camera.height
        im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if im is None:
            logging.error(f"Failed to read image: {src}")
            return False

        img_h, img_w = im.shape[:2]
        if (img_w, img_h) != (w, h):
            logging.error(
                f"Image size mismatch for {img.name}: "
                f"camera=({w}x{h}), file=({img_w}x{img_h})"
            )
            return False
    return True


@typechecked
def undistort_images(
    image_dir: Path, model_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    logging.info("(Optional) Undistorting images...")
    if not (output_dir / "sparse").exists():
        recon = pycolmap.Reconstruction(model_dir)
        is_posed = all(image.has_pose for image in recon.images.values())
        if is_posed:
            pycolmap.undistort_images(output_dir, model_dir, image_dir)
        else:
            # For unposed models, pass image names explicitly
            # since the default path only processes registered
            # (posed) images.
            image_names = [img.name for img in recon.images.values()]
            pycolmap.undistort_images(
                output_dir,
                model_dir,
                image_dir,
                image_names=image_names,
            )
    return output_dir / "images", output_dir / "sparse"


@typechecked
def resize_images_to_max_dim(
    image_dir: Path, model_dir: Path, max_image_dim: int
):
    logging.info("(Optional) Resizing images to maximum dimension...")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    # Load reconstruction and rescale cameras in-place ---
    recon = pycolmap.Reconstruction(str(model_dir))
    any_changed = False
    for cam in recon.cameras.values():
        w, h = int(cam.width), int(cam.height)
        largest = max(w, h)
        scale = min(max_image_dim / largest, 1.0)
        if scale < 1.0:
            cam.rescale(scale)  # scales fx, fy, cx, cy and updates width/height
            any_changed = True

    # If nothing to do, bail early.
    if not any_changed:
        return

    # Persist the updated intrinsics back to the same folder.
    recon.write(str(model_dir))

    # Resize all *registered* images to match their camera's new dims
    for img in recon.images.values():
        rel_path = Path(img.name)
        src = (image_dir / rel_path).resolve()
        if not src.exists():
            logging.fatal("Image does not exist.")

        new_w, new_h = img.camera.width, img.camera.height
        im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if im is None:
            logging.fatal("Image is not readable.")

        h, w = im.shape[:2]
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(
                im, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(src), resized)


@typechecked
def resize_images(
    image_dir: Path,
    recon: pycolmap.Reconstruction,
    max_image_dim: int,
):
    """Resize images and rescale camera intrinsics in-memory.

    Unlike resize_images_to_max_dim, this works with an in-memory
    Reconstruction and does not require model_dir on disk.
    """
    logging.info("(Optional) Resizing images to maximum dimension...")
    any_changed = False
    for cam in recon.cameras.values():
        w, h = int(cam.width), int(cam.height)
        largest = max(w, h)
        scale = min(max_image_dim / largest, 1.0)
        if scale < 1.0:
            cam.rescale(scale)
            any_changed = True

    if not any_changed:
        return

    for img in recon.images.values():
        rel_path = Path(img.name)
        src = (image_dir / rel_path).resolve()
        if not src.exists():
            logging.fatal("Image does not exist.")

        new_w, new_h = img.camera.width, img.camera.height
        im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if im is None:
            logging.fatal("Image is not readable.")

        h, w = im.shape[:2]
        if (w, h) != (new_w, new_h):
            resized = cv2.resize(
                im, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(str(src), resized)


@typechecked
def compute_neighbors(
    point_triangulation_dir: Path, options: PairGenerationOptions
) -> dict[int, list[int]]:
    model = scene.COLMAPMVSModel()
    model.read_from_colmap(str(point_triangulation_dir), "sparse", "images")
    neighbors = scene.compute_neighbors(
        model,
        options.n_neighbors,
        min_triangulation_angle=options.min_triangulation_angle,
        neighbor_type=options.neighbor_type,
    )
    return neighbors


@typechecked
def compute_ranges(
    point_triangulation_dir: Path, options: RangeCalculatorOptions
) -> dict[int, list[int]]:
    model = scene.COLMAPMVSModel()
    model.read_from_colmap(str(point_triangulation_dir), "sparse", "images")
    range_robust = (options.robust_range_min, options.robust_range_max)
    return model.compute_ranges(range_robust, options.stretch_ratio)


@typechecked
def compute_metainfos(
    point_triangulation_dir: Path, options: MetaInfoComputerOptions
) -> tuple[dict[int, list[int]], Ranges]:
    if (options.cache_file is not None) and options.cache_file.exists():
        neighbors, ranges = limapio.read_txt_metainfos(options.cache_file)
        return neighbors, ranges
    model = scene.COLMAPMVSModel()
    model.read_from_colmap(str(point_triangulation_dir), "sparse", "images")
    pairgen_options = options.pair_generation
    neighbors = scene.compute_neighbors(
        model,
        pairgen_options.n_neighbors,
        min_triangulation_angle=pairgen_options.min_triangulation_angle,
        neighbor_type=pairgen_options.neighbor_type,
    )
    rangecal_options = options.range_calculator
    range_robust = (
        rangecal_options.robust_range_min,
        rangecal_options.robust_range_max,
    )
    ranges = model.compute_ranges(range_robust, rangecal_options.stretch_ratio)
    if options.cache_file is not None:
        limapio.save_txt_metainfos(options.cache_file, neighbors, ranges)
    return neighbors, ranges
