from _limap import _vpdet

def AssociateVPs(lines, config=None):
    '''
    Input:
    - lines: list of _base.Line2d
    '''
    if config is None:
        config = _vpdet.VPDetectorConfig()
    detector = _vpdet.VPDetector(config)
    vpresult = detector.AssociateVPs(lines)
    return vpresult

def AssociateVPsParallel(all_lines, config=None):
    '''
    Input:
    - lines: (n_images) list of (list of _base.Line2d)
    '''
    if config is None:
        config = _vpdet.VPDetectorConfig()
    detector = _vpdet.VPDetector(config)
    vpresults = detector.AssociateVPsParallel(all_lines)
    return vpresults

