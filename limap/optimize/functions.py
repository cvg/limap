import numpy as np

import limap.base as _base


def unit_test_add_noise_to_track(track):
    # for unit test
    tmptrack = _base.LineTrack(track)
    start = track.line.start + (np.random.rand(3) - 0.5) * 1e-1
    end = track.line.end + (np.random.rand(3) - 0.5) * 1e-1
    tmpline = _base.Line3d(start, end)
    tmptrack.line = tmpline
    return tmptrack
