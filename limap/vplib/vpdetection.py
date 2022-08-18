from _limap import _vplib

def AssociateVPs(lines, config=None):
    '''
    JLinkage for vp association
    Input:
    - lines: list of _base.Line2d
    '''
    if config is None:
        config = _vplib.JLinkageConfig()
    detector = _vplib.JLinkage(config)
    vpresult = detector.AssociateVPs(lines)
    return vpresult

def AssociateVPsParallel(all_lines, config=None):
    '''
    JLinkage for parallel vp association
    Input:
    - lines: (n_images) list of (list of _base.Line2d)
    '''
    if config is None:
        config = _vplib.JLinkageConfig()
    detector = _vplib.JLinkage(config)
    vpresults = detector.AssociateVPsParallel(all_lines)
    return vpresults

