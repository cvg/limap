import os
from typing import NamedTuple

import joblib
from tqdm import tqdm

import limap.util.io as limapio


class BaseMatcherOptions(NamedTuple):
    """
    Base options for the line matcher

    :param topk: number of top matches for each line \
        (if equal to 0, do mutual nearest neighbor matching)
    :param n_neighbors: number of visual neighbors, \
        only for naming the output folder
    :param n_jobs: number of jobs at multi-processing \
        (please make sure not to exceed the GPU memory limit \
        with learning methods)
    :param weight_path: specify path to load weights \
        (at default, weights will be downloaded to ~/.local)
    """

    topk: int = 10
    n_neighbors: int = 20
    n_jobs: int = 1
    weight_path: str = None


DefaultMatcherOptions = BaseMatcherOptions()


class BaseMatcher:
    """
    Virtual class for line matcher
    """

    def __init__(self, extractor, options=DefaultMatcherOptions):
        self.extractor = extractor
        self.topk = options.topk
        self.n_neighbors = options.n_neighbors
        self.n_jobs = options.n_jobs
        self.weight_path = options.weight_path

    # The functions below are required for matchers
    def get_module_name(self):
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    def match_pair(self, descinfo1, descinfo2):
        """
        Virtual method (need to be implemented) - match two set \
            of lines based on the descriptors
        """
        raise NotImplementedError

    def get_matches_folder(self, output_folder):
        """
        Return the folder path to the output matches

        Args:
            output_folder (str): The output folder
        Returns:
            path_to_matches (str): The path to the saved matches
        """
        return os.path.join(
            output_folder,
            f"{self.get_module_name()}_n{self.n_neighbors}_top{self.topk}",
        )

    def read_descinfo(self, descinfo_folder, idx):
        return self.extractor.read_descinfo(descinfo_folder, idx)

    def get_match_filename(self, matches_folder, idx):
        """
        Return the filename of the matches specified by an image id

        Args:
            matches_folder (str): The output matching folder
            idx (int): image id
        """
        fname = os.path.join(matches_folder, f"matches_{idx}.npy")
        return fname

    def save_match(self, matches_folder, idx, matches):
        """
        Save the output matches from one image to its neighbors

        Args:
            matches_folder (str): The output matching folder
            idx (int): image id
            matches (dict[int -> :class:`np.array`]): \
                The output matches for each neighboring image, \
                each with shape (N, 2)
        """
        fname = self.get_match_filename(matches_folder, idx)
        limapio.save_npy(fname, matches)

    def read_match(self, matches_folder, idx):
        """
        Read the matches for one image with its neighbors

        Args:
            matches_folder (str): The output matching folder
            idx (int): image id
        Returns:
            matches (dict[int -> :class:`np.array`]): \
                The output matches for each neighboring image, \
                each with shape (N, 2)
        """
        fname = self.get_match_filename(matches_folder, idx)
        return limapio.read_npy(fname).item()

    def match_all_neighbors(
        self,
        output_folder,
        image_ids,
        neighbors,
        descinfo_folder,
        skip_exists=False,
    ):
        """
        Match all images with its visual neighbors

        Args:
            output_folder (str): The output folder
            image_ids (list[int]): list of image ids
            neighbors (dict[int -> list[int]]): visual neighbors for each image
            descinfo_folder (str): The folder storing all the descriptors
            skip_exists (bool): Whether to skip already processed images
        Returns:
            matches_folder: The output matching folder
        """
        matches_folder = self.get_matches_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(matches_folder)
        limapio.check_makedirs(matches_folder)

        # multiprocessing unit
        def process(
            self,
            matches_folder,
            descinfo_folder,
            img_id,
            ng_img_id_list,
            skip_exists,
        ):
            fname_save = self.get_match_filename(matches_folder, img_id)
            if skip_exists and os.path.exists(fname_save):
                return
            descinfo1 = self.read_descinfo(descinfo_folder, img_id)
            matches_idx = {}
            for ng_img_id in ng_img_id_list:
                descinfo2 = self.read_descinfo(descinfo_folder, ng_img_id)
                matches = self.match_pair(descinfo1, descinfo2)
                matches_idx.update({ng_img_id: matches})
            self.save_match(matches_folder, img_id, matches_idx)

        joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(process)(
                self,
                matches_folder,
                descinfo_folder,
                img_id,
                neighbors[img_id],
                skip_exists,
            )
            for img_id in tqdm(image_ids)
        )
        return matches_folder

    def match_all_exhaustive_pairs(
        self, output_folder, image_ids, descinfo_folder, skip_exists=False
    ):
        """
        Match all images exhaustively

        Args:
            output_folder (str): The output folder
            image_ids (list[int]): list of image ids
            descinfo_folder (str): The folder storing all the descriptors
            skip_exists (bool): Whether to skip already processed images
        Returns:
            matches_folder: The output matching folder
        """
        matches_folder = self.get_matches_folder(output_folder)
        if not skip_exists:
            limapio.delete_folder(matches_folder)
        limapio.check_makedirs(matches_folder)

        # multiprocessing unit
        def process(
            self,
            matches_folder,
            descinfo_folder,
            img_id,
            ng_img_id_list,
            skip_exists,
        ):
            fname_save = self.get_match_filename(matches_folder, img_id)
            if skip_exists and os.path.exists(fname_save):
                return
            descinfo1 = self.read_descinfo(descinfo_folder, img_id)
            matches_idx = {}
            for ng_img_id in ng_img_id_list:
                if ng_img_id == img_id:
                    continue
                descinfo2 = self.read_descinfo(descinfo_folder, ng_img_id)
                matches = self.match_pair(descinfo1, descinfo2)
                matches_idx.update({ng_img_id: matches})
            self.save_match(matches_folder, img_id, matches_idx)

        joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(process)(
                self,
                matches_folder,
                descinfo_folder,
                img_id,
                image_ids,
                skip_exists,
            )
            for img_id in tqdm(image_ids)
        )
        return matches_folder
