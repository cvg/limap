from .base_vp_detector import BaseVPDetectorOptions


def get_vp_detector(cfg_vp_detector, n_jobs=1):
    """
    Get a vanishing point detector specified by cfg_vp_detector["method"]

    Args:
        cfg_vp_detector: config for the vanishing point detector
    """
    options = BaseVPDetectorOptions()
    options = options._replace(n_jobs=n_jobs)

    method = cfg_vp_detector["method"]
    if method == "jlinkage":
        from .JLinkage import JLinkage

        return JLinkage(cfg_vp_detector, options)
    elif method == "progressivex":
        from .progressivex import ProgressiveX

        return ProgressiveX(cfg_vp_detector, options)
    else:
        raise NotImplementedError
