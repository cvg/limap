import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import pdb
import core.detector.LSD as lsd
import core.detector.SOLD2 as sold2
import core.visualize as vis

def read_list_query(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    image_list = []
    for line in lines:
        imname = line.strip('\n').split(' ')[0]
        image_list.append(imname)
    return image_list

def main():
    basedir = os.path.expanduser('~/data/Localization/Aachen-1.1/undistorted_query')
    fname = os.path.join(basedir, 'list_query.txt')
    query_image_list = read_list_query(fname)
    imname_list = [os.path.join(basedir, k) for k in query_image_list]
    vis.txt_save_image_list(imname_list, 'tmp/0215/query_image_list.txt')
    all_2d_segs = lsd.lsd_detect_2d_segs_on_images(imname_list)
    vis.txt_save_detections(all_2d_segs, "tmp/0215/query_detections.txt")

if __name__ == '__main__':
    main()


