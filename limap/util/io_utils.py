import os, sys
import numpy as np
import shutil
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import limap.base as _base

def check_directory(fname):
    if fname is None:
        raise ValueError("Error! Input directory is None!")
    if not os.path.exists(os.path.dirname(fname)):
        raise ValueError("Error! Base directory {0} does not exist!".format(os.path.dirname(fname)))

def check_path(fname):
    if fname is None:
        raise ValueError("Error! Input filepath is None!")
    if not os.path.exists(fname):
        raise ValueError("Error! File {0} does not exist!".format(fname))

def save_npy(fname, nparray):
    check_directory(fname)
    with open(fname, 'wb') as f:
        np.save(f, nparray)

def read_npy(fname):
    check_path(fname)
    with open(fname, 'rb') as f:
        nparray = np.load(f, allow_pickle=True)
    return nparray

def save_txt_metainfos(fname, neighbors, ranges):
    '''
    Write out .txt for neighbors and ranges
    '''
    check_directory(fname)
    with open(fname, 'w') as f:
        f.write("number of images, {0}\n".format(len(neighbors)))
        f.write("x-range, {0}, {1}\n".format(ranges[0][0], ranges[1][0]))
        f.write("y-range, {0}, {1}\n".format(ranges[0][1], ranges[1][1]))
        f.write("z-range, {0}, {1}\n".format(ranges[0][2], ranges[1][2]))
        for idx, neighbor in enumerate(neighbors):
            str_ = "image {0}".format(idx)
            for ng_idx in neighbor:
                str_ += ", {0}".format(ng_idx)
            f.write(str_ + '\n')

def read_txt_metainfos(fname):
    '''
    Read in .txt for neighbors and ranges
    '''
    check_path(fname)
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(',')[1])
    counter += 1
    ranges = (np.zeros(3), np.zeros(3))
    k = txt_lines[counter].strip().split(',')[1:]
    ranges[0][0], ranges[1][0] = float(k[0]), float(k[1])
    counter += 1
    k = txt_lines[counter].strip().split(',')[1:]
    ranges[0][1], ranges[1][1] = float(k[0]), float(k[1])
    counter += 1
    k = txt_lines[counter].strip().split(',')[1:]
    ranges[0][2], ranges[1][2] = float(k[0]), float(k[1])
    counter += 1
    neighbors = []
    for img_id in range(n_images):
        k = txt_lines[counter].strip().split(',')[1:]
        neighbor = [int(kk) for kk in k]
        neighbors.append(neighbor)
        counter += 1
    return neighbors, ranges

def save_txt_imname_list(fname, imname_list):
    check_directory(fname)
    with open(fname, 'w') as f:
        f.write('number of images, {0}\n'.format(len(imname_list)))
        for imname in imname_list:
            f.write(imname + '\n')

def read_txt_imname_list(fname):
    check_path(fname)
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(',')[1])
    counter += 1
    imname_list = []
    for img_id in range(n_images):
        imname = txt_lines[counter].strip()
        imname_list.append(imname)
        counter += 1
    return imname_list

def save_obj(fname, lines):
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

def save_txt_linetracks(fname, linetracks, n_visible_views=4):
    '''
    Save all the linetracks into a single .txt file.
    '''
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    linetracks = [track for track in linetracks if track.count_images() >= n_visible_views]
    print("Writing all linetracks to a single file...")
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

def save_folder_linetracks(folder, linetracks):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    print("Writing linetracks to {0}...".format(folder))
    n_tracks = len(linetracks)
    for track_id in tqdm(range(n_tracks)):
        fname = os.path.join(folder, 'track_{0}.txt'.format(track_id))
        linetracks[track_id].Write(fname)

def read_folder_linetracks(folder):
    check_path(folder)
    flist = os.listdir(folder)
    n_tracks = 0
    for fname in flist:
        if fname[-4:] == '.txt' and fname[:5] == 'track':
            n_tracks += 1
    linetracks = []
    for track_id in range(n_tracks):
        fname = os.path.join(folder, 'track_{0}.txt'.format(track_id))
        track = _base.LineTrack()
        track.Read(fname)
        linetracks.append(track)
    return linetracks


