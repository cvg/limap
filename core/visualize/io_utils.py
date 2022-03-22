import os, sys
import numpy as np
import shutil
from tqdm import tqdm

def save_npy(fname, lines, counts=None, img_hw=None):
    with open(fname, 'wb') as f:
        np.savez(f, lines=lines, counts=counts, img_hw=img_hw)

def load_npy(fname):
    with open(fname, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        lines, counts = data["lines"], data["counts"]
    return lines, counts

def save_obj(fname, lines, counts=None, n_visible_views=4):
    # save obj for CloudCompare visualization
    if type(lines) == list:
        if type(lines[0]) == np.ndarray:
            lines = np.array(lines)
        else:
            lines = np.array([line.as_array() for line in lines])
    n_lines = lines.shape[0]
    vertices = []
    for line_idx in range(n_lines):
        line = lines[line_idx]
        if counts is not None:
            count = counts[line_idx]
            if count < n_visible_views:
                continue
        vertices.append(line[0])
        vertices.append(line[1])
    n_lines = int(len(vertices) / 2)
    with open(fname, 'w') as f:
        for v in vertices:
            f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
        for idx in range(n_lines):
            f.write('l {0} {1}\n'.format(2*idx+1, 2*idx+2))

def load_obj(fname):
    with open(fname, 'r') as f:
        flines = f.readlines()
    counter = 0
    vertices = []
    while True:
        fline = flines[counter].strip('\n').split(' ')
        if fline[0] != 'v':
            break
        vertice = np.array([float(fline[1]), float(fline[2]), float(fline[3])])
        vertices.append(vertice)
        counter += 1
    vertices = np.array(vertices)
    n_lines = int(vertices.shape[0] / 2)
    lines = vertices.reshape((n_lines, 2, 3))
    return lines

def tmp_save_all_2d_segs_for_l3dpp(imname_list, all_2d_segs, img_hw, folder="tmp/l3dpp"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    img_h, img_w = img_hw[0], img_hw[1]
    n_images = len(all_2d_segs)
    # TODO now it is hard-coded here
    mode = 'default'
    if os.path.basename(imname_list[0])[0] == '0': # tnt
        mode = 'tnt'
        number_list = [int(os.path.basename(imname)[:-4]) for imname in imname_list]
        index_list = np.argsort(number_list).tolist()
    for idx in range(n_images):
        if mode == 'default':
            image_id = idx + 1
        elif mode == 'tnt':
            image_id = index_list.index(idx) + 1
        else:
            raise NotImplementedError
        fname = "segments_L3D++_{0}_{1}x{2}_3000.txt".format(image_id, img_w, img_h)
        fname = os.path.join(folder, fname)
        segs = all_2d_segs[idx]
        n_segments = segs.shape[0]
        with open(fname, 'w') as f:
            f.write("{0}\n".format(n_segments))
            for line_id in range(n_segments):
                line = segs[line_id]
                f.write("{0} {1} {2} {3}\n".format(line[0], line[1], line[2], line[3]))
        print("Writing for L3DPP: {0}".format(fname))

def read_all_2d_segs_from_l3dpp(folder):
    import pdb
    flist = os.listdir(folder)
    n_images = len(flist)
    all_2d_segs = []
    for idx in range(n_images):
        fname = os.path.join(folder, 'segments_L3D++_{0}_800x600_3000.txt'.format(idx+1))
        with open(fname, 'r') as f:
            lines = f.readlines()
        n_segs = int(lines[0].strip('\n'))
        segs = []
        for seg_id in range(n_segs):
            line = lines[seg_id + 1].strip('\n').split(' ')
            line = [float(k) for k in line]
            segs.append(line)
        segs = np.array(segs)
        all_2d_segs.append(segs)
    return all_2d_segs

def save_datalist_to_folder(folder, prefix, imname_list, arrlist, is_descinfo=False):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    assert len(imname_list) == len(arrlist)
    with open(os.path.join(folder, 'imname_list.npy'), 'wb') as f:
        np.savez(f, imname_list=imname_list)
    n_images = len(imname_list)
    print("Writing to {0}...".format(folder))
    for img_id in tqdm(range(n_images)):
        fname = os.path.join(folder, '{0}_{1}.npy'.format(prefix, img_id))
        with open(fname, 'wb') as f:
            arr = arrlist[img_id]
            if type(arr) == list: # for descinfos
                if len(arr) == 2 and is_descinfo:
                    arr[1] = arr[1][None,...]
            np.savez(f, data=arr)
            if type(arr) == list: # for descinfos
                if len(arr) == 2 and is_descinfo:
                    arr[1] = arr[1][0]

def load_datalist_from_folder(folder, prefix):
    if not os.path.exists(folder):
        raise ValueError("Error! Path {0} does not exist".format(folder))
    with open(os.path.join(folder, 'imname_list.npy'), 'rb') as f:
        data = np.load(f, allow_pickle=True)
        imname_list = data["imname_list"]
    n_images = len(imname_list)
    arrlist = []
    print("Loading {0}...".format(folder))
    for img_id in tqdm(range(n_images)):
        fname = os.path.join(folder, '{0}_{1}.npy'.format(prefix, img_id))
        with open(fname, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            arr = data["data"]
            if len(arr) == 2: # for descinfos
                if len(arr[1].shape) == 3:
                    arr[1] = arr[1][0]
            arrlist.append(arr)
    return arrlist

def save_linetracks_to_folder(linetracks, folder):
    if os.path.exists(folder):
        cmd = "rm {0}/track_*.txt".format(folder)
        os.system(cmd)
    else:
        os.makedirs(folder)
    print("Writing linetracks to {0}...".format(folder))
    n_tracks = len(linetracks)
    for track_id in tqdm(range(n_tracks)):
        fname = os.path.join(folder, 'track_{0}.txt'.format(track_id))
        linetracks[track_id].Write(fname)

def count_linetracks_from_folder(folder):
    if not os.path.exists(folder):
        raise ValueError("Error! Path {0} does not exist".format(folder))
    flist = os.listdir(folder)
    counter = 0
    for fname in flist:
        if fname[-4:] == '.txt':
            counter += 1
    return counter

def load_linetracks_from_folder(linetracks, folder):
    '''
    -Input:
    linetracks: list of empty _base.LineTrack object
    '''
    n_tracks = count_linetracks_from_folder(folder)
    assert n_tracks == len(linetracks)
    print("Loading linetracks from {0}...".format(folder))
    for track_id in range(n_tracks):
        fname = os.path.join(folder, 'track_{0}.txt'.format(track_id))
        linetracks[track_id].Read(fname)
    return linetracks

def txt_save_image_list(imname_list, fname):
    n_images = len(imname_list)
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_images))
        for idx, imname in enumerate(imname_list):
            f.write("{0} {1}\n".format(idx, imname))

def txt_save_cameras(cameras, fname):
    n_cams = len(cameras)
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_cams))
        for idx, cam in enumerate(cameras):
            f.write("{0} {1} {2} {3} {4} {5} {6}\n".format(idx, cam.w, cam.h, cam.K[0,0], cam.K[1,1], cam.K[0,2], cam.K[1,2]))
            for row_id in range(3):
                f.write("{0} {1} {2} {3}\n".format(cam.R[row_id, 0], cam.R[row_id, 1], cam.R[row_id, 2], cam.T[row_id]))

def txt_save_neighbors(neighbors, fname):
    n_images = len(neighbors)
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_images))
        for img_id, neighbor in enumerate(neighbors):
            f.write("{0} ".format(img_id))
            for ng_img_id in neighbor:
                f.write("{0} ".format(ng_img_id))
            f.write("\n")

def txt_save_detections(all_2d_segs, fname):
    print("Writing all detections to a single file...")
    n_images = len(all_2d_segs)
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_images))
        for img_id, segs in enumerate(tqdm(all_2d_segs)):
            n_lines = segs.shape[0]
            f.write("{0} {1}\n".format(img_id, n_lines))
            for line_id in range(n_lines):
                f.write("{0:.10f} {1:.10f} {2:.10f} {3:.10f}\n".format(segs[line_id][0], segs[line_id][1], segs[line_id][2], segs[line_id][3]))


def txt_save_linetracks(linetracks, fname, n_visible_views=4):
    linetracks = [track for track in linetracks if track.count_images() >= n_visible_views]
    print("Writing all linetracks to a single file...")
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    n_tracks = len(linetracks)
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_tracks))
        for track_id in tqdm(range(n_tracks)):
            track = linetracks[track_id]
            f.write("{0} {1} {2}\n".format(track_id, track.count_lines(), track.count_images()))
            f.write("{0:.10f} {1:.10f} {2:.10f}\n".format(track.line.start[0], track.line.start[1], track.line.start[2]))
            f.write("{0:.10f} {1:.10f} {2:.10f}\n".format(track.line.end[0], track.line.end[1], track.line.end[2]))
            for idx in range(track.count_lines()):
                f.write("{0} ".format(track.image_id_list[idx]))
            f.write("\n")
            for idx in range(track.count_lines()):
                f.write("{0} ".format(track.line_id_list[idx]))
            f.write("\n")


