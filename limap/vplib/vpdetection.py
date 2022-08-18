from _limap import _vplib

def AssociateVPs(lines, config=None):
    '''
    Input:
    - lines: list of _base.Line2d
    '''
    if config is None:
        config = _vplib.VPDetectorConfig()
    detector = _vplib.VPDetector(config)
    vpresult = detector.AssociateVPs(lines)
    return vpresult

def AssociateVPsParallel(all_lines, config=None):
    '''
    Input:
    - lines: (n_images) list of (list of _base.Line2d)
    '''
    if config is None:
        config = _vplib.VPDetectorConfig()
    detector = _vplib.VPDetector(config)
    vpresults = detector.AssociateVPsParallel(all_lines)
    return vpresults

