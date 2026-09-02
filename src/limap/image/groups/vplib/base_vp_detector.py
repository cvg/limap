import joblib
from tqdm import tqdm
from pathlib import Path
from typeguard import typechecked

import limap.geometry
from limap._limap._image._groups._vplib import VPResult, BaseVPDetectorOptions

DefaultVPDetectorOptions = BaseVPDetectorOptions()


class BaseVPDetector:
    def __init__(self, options=DefaultVPDetectorOptions):
        self.n_jobs = options.n_jobs

    # Module name needs to be set
    def get_module_name(self) -> str:
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    # The functions below are required for VP detectors
    @typechecked
    def detect_vp(
        self, lines: list[limap.geometry.Line2d], image_path: Path | None = None
    ) -> VPResult:
        """
        Virtual method (need to be implemented) - detect vanishing points
        """
        raise NotImplementedError

    @typechecked
    def detect_vp_all_images(
        self,
        all_lines: dict[int, list[limap.geometry.Line2d]],
        image_paths: dict[int, Path] | None = None,
    ) -> dict[int, VPResult]:
        """
        Detect vanishing points on multiple images with multiple processes
        """

        def process(self, lines):
            return self.detect_vp(lines)

        def process_image_path(self, lines, image_path):
            return self.detect_vp(lines, image_path)

        if image_paths is None:
            vpresults_vector = joblib.Parallel(self.n_jobs)(
                joblib.delayed(process)(self, lines)
                for (img_id, lines) in tqdm(all_lines.items())
            )
        else:
            assert len(all_lines) == len(image_paths)
            vpresults_vector = joblib.Parallel(self.n_jobs)(
                joblib.delayed(process_image_path)(
                    self, lines, image_paths[img_id]
                )
                for (img_id, lines) in tqdm(all_lines.items())
            )
        # map vector back to map
        vpresults = dict()
        for idx, img_id in enumerate(list(all_lines.keys())):
            vpresults[img_id] = vpresults_vector[idx]
        return vpresults

    def visualize(
        self, fname, img, lines, vpresult, show_original=False, endpoints=False
    ):
        import cv2

        import limap.visualize as limapvis

        img = limapvis.draw_2d_vpresult(
            img,
            lines,
            vpresult,
            show_original=show_original,
            endpoints=endpoints,
        )
        cv2.imwrite(fname, img)
