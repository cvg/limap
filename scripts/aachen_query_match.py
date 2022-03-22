import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py
import pdb
import core.detector.SOLD2 as sold2

def read_lines_from_h5(fname):
    f = h5py.File(fname, 'r')
    m = {}
    # db folder
    for key in f['db'].keys():
        fname = os.path.join('db', key)
        lines = np.array(f['db'][key]['lines'])
        m[fname] = lines
    # sequencec folder
    for key1 in f['sequences'].keys():
        path = os.path.join('sequences', key1)
        for key2 in f['sequences'][key1].keys():
            if key2.endswith('.png'):
                fname = os.path.join(path, key2)
                lines = np.array(f['sequences'][key1][key2]['lines'])
                m[fname] = lines
            else:
                for key3 in f['sequences'][key1][key2].keys():
                    fname = os.path.join(path, key2, key3)
                    lines = np.array(f['sequences'][key1][key2][key3]['lines'])
                    m[fname] = lines
    # query
    for key1 in f['query'].keys():
        path = os.path.join('query', key1)
        for key2 in f['query'][key1].keys():
            path = os.path.join(path, key2)
            for key3 in f['query'][key1][key2].keys():
                fname = os.path.join(path, key3)
                lines = np.array(f['query'][key1][key2][key3]['lines'])
                m[fname] = lines
    return m

def main():
    h5_fname = 'format/feats-lsd-r1024.h5'
    m = read_lines_from_h5(h5_fname)
    query_image_list = []
    all_2d_segs = []
    for imname in m.keys():
        if imname[:5] == 'query':
            query_image_list.append(imname)
            all_2d_segs.append(m[imname])

    with open("tmp/query_image_list.npy", "wb") as f:
        np.savez(f, imname_list=query_image_list)
    basedir = os.path.expanduser('~/data/Localization/Aachen-1.1/undistorted_query')
    imname_list = [os.path.join(basedir, k) for k in query_image_list]
    sold2.sold2_compute_descinfos(imname_list, all_2d_segs, descinfo_dir='tmp/query_lsd_descinfos')

    pdb.set_trace()
    all_lines = []
    for imname in imname_list:
        lines = m[imname]
        all_lines.append(lines)
    with open("all_2d_segs.npy", 'wb') as f:
        np.savez(f, all_2d_segs=all_lines)
    pdb.set_trace()

if __name__ == '__main__':
    main()


