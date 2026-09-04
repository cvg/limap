import os

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typeguard import typechecked
from typing import Any
from tqdm import tqdm

import limap.util.io as limapio
import limap.visualize as limapvis


@dataclass
class BaseDetectorOptions:
    """
    Base options for the line detector

    :param set_gray: whether to set the image to gray scale \
        (sometimes depending on the detector)
    :param max_num_2d_segs: maximum number of detected \
        line segments (default = 3000)
    :param do_merge_lines: whether to merge close similar \
        lines at post-processing (default = False)
    :param visualize: whether to output visualizations into \
        output folder along with the detections (default = False)
    :param weight_path: root directory to load/store weights \
        (at default, ``~/.cache/limap``, overridable with the \
        ``LIMAP_WEIGHTS_PATH`` environment variable)
    """

    set_gray: bool = True
    max_num_2d_segs: int = 3000
    do_merge_lines: bool = False
    visualize: bool = False
    weight_path: Path | None = None


DefaultDetectorOptions = BaseDetectorOptions()


class BaseDetector:
    """
    Virtual class for line detector
    """

    def __init__(self, options=DefaultDetectorOptions):
        self.set_gray = options.set_gray
        self.max_num_2d_segs = options.max_num_2d_segs
        self.do_merge_lines = options.do_merge_lines
        self.visualize = options.visualize
        self.weight_path = options.weight_path

    # Module name needs to be set
    @typechecked
    def get_module_name(self) -> str:
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    # The functions below are required for detectors
    @typechecked
    def detect(self, image_path: Path) -> np.ndarray:
        """
        Virtual method (for detector) - detect 2D line segments

        Args:
            image_path: full path to the image
        Returns:
            :class:`np.array` of shape (N, 5): line detections. \
                Each row corresponds to x1, y1, x2, y2 and score.
        """
        raise NotImplementedError

    # The functions below are required for extractors
    @typechecked
    def extract(self, image_path: Path, segs: np.ndarray) -> Any:
        """
        Virtual method (for extractor) - \
        extract the features for the detected segments

        Args:
            image_path: full path to the image
            segs: :class:`np.array` of shape (N, 5), line detections. \
                Each row corresponds to x1, y1, x2, y2 and score. \
                Computed from the `detect` method.
        Returns:
            The extracted feature
        """
        raise NotImplementedError

    @typechecked
    def get_descinfo_fname(self, descinfo_folder: Path, img_id: int) -> Path:
        """
        Virtual method (for extractor) - \
        Get the target filename of the extracted feature

        Args:
            descinfo_folder (pathlib.Path): The output folder
            img_id (int): The image id
        Returns:
            pathlib.Path: target filename
        """
        raise NotImplementedError

    @typechecked
    def save_descinfo(
        self, descinfo_folder: Path, img_id: int, descinfo: Any
    ) -> None:
        """
        Virtual method (for extractor) - \
        Save the extracted feature to the target folder

        Args:
            descinfo_folder (pathlib.Path): The output folder
            img_id (int): The image id
            descinfo: The features extracted from the function `extract`
        """
        raise NotImplementedError

    @typechecked
    def read_descinfo(self, descinfo_folder: Path, img_id: Any) -> Any:
        """
        Virtual method (for extractor) - Read in the extracted feature. \
        Dual function for `save_descinfo`.

        Args:
            descinfo_folder (pathlib.Path): The output folder
            img_id (int): The image id
        Returns:
            The extracted feature
        """
        raise NotImplementedError

    # The functions below are required for double-functioning objects
    @typechecked
    def detect_and_extract(self, image_path: Path) -> tuple[np.ndarray, Any]:
        """
        Virtual method (for dual-functional class that can perform both \
        detection and extraction) - Detect and extract on a single image

        Args:
            image_path: full path to the image
        Returns:
            segs (:class:`np.array`): of shape (N, 5), line detections. \
                Each row corresponds to x1, y1, x2, y2 and score. \
                Computed from the `detect` method.
            descinfo: The features extracted from the function `extract`
        """
        raise NotImplementedError

    @typechecked
    def sample_descinfo_by_indexes(
        self, descinfo: Any, indexes: list[int]
    ) -> Any:
        """
        Virtual method (for dual-functional class that can perform \
        both detection and extraction) -  \
            sample descriptors for a subset of images

        Args:
            descinfo: The features extracted from the function `extract`.
            indexes (list[int]): List of image ids for the subset.
        """
        raise NotImplementedError

    @typechecked
    def get_segments_folder(self, output_folder: Path) -> Path:
        """
        Return the folder path to the detected segments

        Args:
            output_folder (pathlib.Path): The output folder
        Returns:
            path_to_segments (pathlib.Path): The path to the saved segments
        """
        return output_folder / "segments"

    @typechecked
    def get_descinfo_folder(self, output_folder: Path) -> Path:
        """
        Return the folder path to the extracted descriptors

        Args:
            output_folder (pathlib.Path): The output folder
        Returns:
            path_to_descinfos (pathlib.Path): The path to the saved descriptors
        """
        return output_folder / "descinfos" / self.get_module_name()

    def merge_lines(self, segs):
        from .line_utils import merge_lines

        segs = segs[:, :4].reshape(-1, 2, 2)
        segs = merge_lines(segs)
        segs = segs.reshape(-1, 4)
        return segs

    def take_longest_k(self, segs, max_num_2d_segs=3000):
        indexes = np.arange(0, segs.shape[0])
        if max_num_2d_segs is None or max_num_2d_segs == -1:
            pass
        elif segs.shape[0] > max_num_2d_segs:
            lengths_squared = (segs[:, 2] - segs[:, 0]) ** 2 + (
                segs[:, 3] - segs[:, 1]
            ) ** 2
            indexes = np.argsort(lengths_squared)[::-1][:max_num_2d_segs]
            segs = segs[indexes, :]
        return segs, indexes

    def visualize_segs(self, output_folder, image_paths, first_k=10):
        seg_folder = self.get_segments_folder(output_folder)
        min(first_k, len(image_paths))
        vis_folder = os.path.join(output_folder, "visualize")
        limapio.check_makedirs(vis_folder)
        for img_id, image_path in image_paths.items():
            img = cv2.imread(image_path)
            segs = limapio.read_txt_segments(seg_folder, img_id)
            img = limapvis.draw_2d_lines(img, segs, (0, 255, 0))
            fname = os.path.join(vis_folder, f"img_{img_id}_det.png")
            cv2.imwrite(fname, img)

    def detect_all_images(self, output_folder, image_paths, skip_exists=False):
        """
        Perform line detection on all images and save the line segments

        Args:
            output_folder (str): The output folder
            image_paths (dict[int, pathlib.Path]): \
                mapping from image id to full image path
            skip_exists (bool): Whether to skip already processed images
        Returns:
            dict[int -> :class:`np.array`]: \
                The line detection for each image indexed by the image id. \
                Each segment is with shape (N, 5). \
                Each row corresponds to x1, y1, x2, y2 and score.
        """
        seg_folder = self.get_segments_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(seg_folder)
        limapio.check_makedirs(seg_folder)
        if self.visualize:
            vis_folder = os.path.join(output_folder, "visualize")
            limapio.check_makedirs(vis_folder)
        for img_id, image_path in tqdm(image_paths.items()):
            if skip_exists and limapio.exists_txt_segments(seg_folder, img_id):
                if self.visualize:
                    segs = limapio.read_txt_segments(seg_folder, img_id)
            else:
                segs = self.detect(image_path)
                if self.do_merge_lines:
                    segs = self.merge_lines(segs)
                segs, _ = self.take_longest_k(
                    segs, max_num_2d_segs=self.max_num_2d_segs
                )
                limapio.save_txt_segments(seg_folder, img_id, segs)
            if self.visualize:
                img = cv2.imread(image_path)
                img = limapvis.draw_2d_lines(img, segs, (0, 255, 0))
                fname = os.path.join(vis_folder, f"img_{img_id}_det.png")
                cv2.imwrite(fname, img)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        all_2d_segs = {img_id: all_2d_segs[img_id] for img_id in image_paths}
        return all_2d_segs

    def extract_all_images(
        self, output_folder, image_paths, all_2d_segs, skip_exists=False
    ):
        """
        Line descriptor extraction on all images and save the descriptors.

        Args:
            output_folder (str): The output folder.
            image_paths (dict[int, pathlib.Path]): \
                mapping from image id to full image path
            all_2d_segs (dict[int -> :class:`np.array`]): \
                The line detection for each image indexed by the image id. \
                Each segment is with shape (N, 5). \
                Each row corresponds to x1, y1, x2, y2 and score. \
                Computed from `detect_all_images`
            skip_exists (bool): Whether to skip already processed images.
        Returns:
            descinfo_folder (str): The path to the saved descriptors.
        """
        descinfo_folder = self.get_descinfo_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(descinfo_folder)
        limapio.check_makedirs(descinfo_folder)
        for img_id, image_path in tqdm(image_paths.items()):
            if skip_exists and os.path.exists(
                self.get_descinfo_fname(descinfo_folder, img_id)
            ):
                continue
            descinfo = self.extract(image_path, all_2d_segs[img_id])
            self.save_descinfo(descinfo_folder, img_id, descinfo)
        return descinfo_folder

    def detect_and_extract_all_images(
        self, output_folder, image_paths, skip_exists=False
    ):
        """
        Perform line detection and description on all images and \
        save the line segments and descriptors

        Args:
            output_folder (str): The output folder
            image_paths (dict[int, pathlib.Path]): \
                mapping from image id to full image path
            skip_exists (bool): Whether to skip already processed images
        Returns:
            all_segs (dict[int -> :class:`np.array`]): \
                The line detection for each image indexed by the image id. \
                Each segment is with shape (N, 5). \
                Each row corresponds to x1, y1, x2, y2 and score.
            descinfo_folder (str): Path to the extracted descriptors.
        """
        assert not self.do_merge_lines
        seg_folder = self.get_segments_folder(output_folder)
        descinfo_folder = self.get_descinfo_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(seg_folder)
            limapio.delete_folder(descinfo_folder)
        limapio.check_makedirs(seg_folder)
        limapio.check_makedirs(descinfo_folder)
        if self.visualize:
            vis_folder = os.path.join(output_folder, "visualize")
            limapio.check_makedirs(vis_folder)
        for img_id, image_path in tqdm(image_paths.items()):
            if (
                skip_exists
                and os.path.exists(
                    self.get_descinfo_fname(descinfo_folder, img_id)
                )
                and limapio.exists_txt_segments(seg_folder, img_id)
            ):
                if self.visualize:
                    segs = limapio.read_txt_segments(seg_folder, img_id)
            else:
                segs, descinfo = self.detect_and_extract(image_path)
                n_segs_orig = segs.shape[0]
                segs, indexes = self.take_longest_k(
                    segs, max_num_2d_segs=self.max_num_2d_segs
                )
                if indexes.shape[0] < n_segs_orig:
                    descinfo = self.sample_descinfo_by_indexes(
                        descinfo, indexes
                    )
                limapio.save_txt_segments(seg_folder, img_id, segs)
                self.save_descinfo(descinfo_folder, img_id, descinfo)
            if self.visualize:
                img = cv2.imread(image_path)
                img = limapvis.draw_2d_lines(img, segs, (0, 255, 0))
                fname = os.path.join(vis_folder, f"img_{img_id}_det.png")
                cv2.imwrite(fname, img)
        all_2d_segs = limapio.read_all_segments_from_folder(seg_folder)
        all_2d_segs = {img_id: all_2d_segs[img_id] for img_id in image_paths}
        return all_2d_segs, descinfo_folder
