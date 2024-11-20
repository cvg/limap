from typing import NamedTuple

import joblib
from tqdm import tqdm


class BaseVPDetectorOptions(NamedTuple):
    """
    Base options for the vanishing point detector

    :param n_jobs: number of jobs at multi-processing (please make sure \
                   not to exceed the GPU memory limit with learning methods)
    """

    n_jobs: int = 1


DefaultVPDetectorOptions = BaseVPDetectorOptions()


class BaseVPDetector:
    def __init__(self, options=DefaultVPDetectorOptions):
        self.n_jobs = options.n_jobs

    # Module name needs to be set
    def get_module_name(self):
        """
        Virtual method (need to be implemented) - return the name of the module
        """
        raise NotImplementedError

    # The functions below are required for VP detectors
    def detect_vp(self, lines, camview=None):
        """
        Virtual method - detect vanishing points

        Args:
            lines (list[:class:`limap.base.Line2d`]): list of input 2D lines.
            camview (:class:`limap.base.CameraView`): optional, \
                the `limap.base.CameraView` instance corresponding to the image.
        Returns:
            vpresult  type: list[:class:`limap.vplib.VPResult`]
        """
        raise NotImplementedError

    def detect_vp_all_images(self, all_lines, camviews=None):
        """
        Detect vanishing points on multiple images with multiple processes

        Args:
            all_lines (dict[int, list[:class:`limap.base.Line2d`]]): \
                map storing all the lines for each image
            camviews (dict[int, :class:`limap.base.CameraView`]): \
                optional, the `limap.base.CameraView` instances, \
                each corresponding to one image
        """

        def process(self, lines):
            return self.detect_vp(lines)

        def process_camview(self, lines, camview):
            return self.detect_vp(lines, camview)

        if camviews is None:
            vpresults_vector = joblib.Parallel(self.n_jobs)(
                joblib.delayed(process)(self, lines)
                for (img_id, lines) in tqdm(all_lines.items())
            )
        else:
            vpresults_vector = joblib.Parallel(self.n_jobs)(
                joblib.delayed(process_camview)(self, lines, camviews[img_id])
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

        img = limapvis.vis_vpresult(
            img,
            lines,
            vpresult,
            show_original=show_original,
            endpoints=endpoints,
        )
        cv2.imwrite(fname, img)
