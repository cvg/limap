import os, sys
import numpy as np

import core.visualize as vis
import limap.base as _base

import pdb

n_visible_views = 4
basedir = '/media/shaoliu/cvg-hdd-usb-2tb-08/Lines/experiments/limap_tnt_lsd'
expdir = os.path.join(basedir, "Truck")
input_dir = os.path.join(expdir, "finaltracks")

def main():
    # load linetracks
    n_tracks = vis.count_linetracks_from_folder(input_dir)
    linetracks = [_base.LineTrack() for idx in range(n_tracks)]
    linetracks = vis.load_linetracks_from_folder(linetracks, input_dir)

    # simulate timer
    thresholds = []
    for idx, track in enumerate(linetracks):
        line2d_list = track.line2d_list
        if len(line2d_list) < n_visible_views:
            thresholds.append(-1)
            continue
        lengths = [line2d.length() for line2d in line2d_list]
        th = np.sort(lengths)[-n_visible_views]
        thresholds.append(th)

    thresholds = np.array(thresholds).astype(int)
    thresholds[thresholds > 400] = 400
    thresholds[thresholds > 0] = thresholds[thresholds > 0] / 2.0
    thresholds[thresholds != -1] = 200 - thresholds[thresholds != -1]

    fname_save = os.path.join(expdir, 'timers_threshold.npy')
    np.save(fname_save, thresholds)
    pdb.set_trace()

main()

