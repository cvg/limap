import os
import numpy as np
import shutil
from tqdm import tqdm
import limap.base as _base

def check_directory(fname):
    if fname is None:
        raise ValueError("Error! Input directory is None!")
    if os.path.dirname(fname) != '' and (not os.path.exists(os.path.dirname(fname))):
        raise ValueError("Error! Base directory {0} does not exist!".format(os.path.dirname(fname)))

def check_path(fname):
    if fname is None:
        raise ValueError("Error! Input filepath is None!")
    if not os.path.exists(fname):
        raise ValueError("Error! File {0} does not exist!".format(fname))

def delete_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

def check_makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_npy(fname, nparray):
    check_directory(fname)
    with open(fname, 'wb') as f:
        np.save(f, np.array(nparray, dtype=object))

def read_npy(fname):
    check_path(fname)
    with open(fname, 'rb') as f:
        nparray = np.load(f, allow_pickle=True)
    return nparray

def save_npz(fname, dic):
    check_directory(fname)
    np.savez(fname, **dic)

def read_npz(fname):
    check_path(fname)
    return np.load(fname, allow_pickle=True)

def save_ply(fname, points):
    from plyfile import PlyData, PlyElement
    points = np.array(points)
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(fname)

def read_ply(fname):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points = np.stack([x, y, z], axis=1)
    print("number of points: {0}".format(points.shape[0]))
    return points

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
        for img_id, neighbor in neighbors.items():
            str_ = "image {0}".format(img_id)
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
    neighbors = {}
    for idx in range(n_images):
        k = txt_lines[counter].strip().split(',')
        img_id = int(k[0][6:])
        neighbor = [int(kk) for kk in k[1:]]
        neighbors[img_id] = neighbor
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

def save_txt_imname_dict(fname, imname_dict):
    check_directory(fname)
    with open(fname, 'w') as f:
        f.write('number of images, {0}\n'.format(len(imname_dict)))
        for img_id, imname in imname_dict.items():
            f.write("{0}, {1}\n".format(img_id, imname))

def read_txt_imname_dict(fname):
    check_path(fname)
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(',')[1])
    counter += 1
    imname_dict = {}
    for img_id in range(n_images):
        line = txt_lines[counter].strip().split(",")
        img_id, imname = int(line[0]), line[1][1:]
        imname_dict[img_id] = imname
        counter += 1
    return imname_dict

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

def save_l3dpp(folder, imagecols, all_2d_segs):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    n_images = len(all_2d_segs)
    assert imagecols.NumImages() == len(all_2d_segs)
    image_names = imagecols.get_image_name_list()

    # TODO make this function general for different input resolution within the set
    first_cam = imagecols.cam(imagecols.get_cam_ids()[0])
    height, width = first_cam.h(), first_cam.w()

    # TODO now it is hard-coded here (need to deal with the weird id mapping of Line3D++)
    mode = 'default'
    if os.path.basename(image_names[0])[0] == '0': # tnt
        mode = 'tnt'
        number_list = [int(os.path.basename(imname)[:-4]) for imname in image_names]
        index_list = np.argsort(number_list).tolist()
    for idx in imagecols.get_img_ids():
        if mode == 'default':
            image_id = idx
        elif mode == 'tnt':
            image_id = index_list.index(idx)
        else:
            raise NotImplementedError
        fname = "segments_L3D++_{0}_{1}x{2}_3000.txt".format(image_id, width, height)
        fname = os.path.join(folder, fname)
        segs = all_2d_segs[idx]
        n_segments = segs.shape[0]
        with open(fname, 'w') as f:
            f.write("{0}\n".format(n_segments))
            for line_id in range(n_segments):
                line = segs[line_id]
                f.write("{0} {1} {2} {3}\n".format(line[0], line[1], line[2], line[3]))
        print("Writing for L3DPP: {0}".format(fname))

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
    print("Read linetracks from {0}...".format(folder))
    linetracks = []
    for track_id in range(n_tracks):
        fname = os.path.join(folder, 'track_{0}.txt'.format(track_id))
        track = _base.LineTrack()
        track.Read(fname)
        linetracks.append(track)
    return linetracks

def save_folder_linetracks_with_info(folder, linetracks, config=None, imagecols=None, all_2d_segs=None):
    save_folder_linetracks(folder, linetracks)
    if config is not None:
        save_npy(os.path.join(folder, "config.npy"), config)
    if imagecols is not None:
        save_npy(os.path.join(folder, "imagecols.npy"), imagecols.as_dict())
    if all_2d_segs is not None:
        save_npy(os.path.join(folder, "all_2d_segs.npy"), all_2d_segs)

def read_folder_linetracks_with_info(folder):
    linetracks = read_folder_linetracks(folder)
    cfg, imagecols, all_2d_segs = None, None, None
    if os.path.isfile(os.path.join(folder, "config.npy")):
        cfg = read_npy(os.path.join(folder, "config.npy"))
    if os.path.isfile(os.path.join(folder, "imagecols.npy")):
        imagecols = _base.ImageCollection(read_npy(os.path.join(folder, "imagecols.npy")).item())
    if os.path.isfile(os.path.join(folder, "all_2d_segs.npy")):
        all_2d_segs = read_npy(os.path.join(folder, "all_2d_segs.npy"))
    return linetracks, cfg.item(), imagecols, all_2d_segs.item()

def read_txt_Line3Dpp(fname):
    linetracks = []
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    line_counts = []
    line_track_id_list = []
    line_counters = 0
    for txt_line in txt_lines:
        txt_line = txt_line.strip('\n').split(' ')
        counter = 0
        n_lines = int(txt_line[counter])
        counter += 1
        line_counters += n_lines
        # get line 3d
        line3d_list = []
        for idx in range(n_lines):
            infos = [float(k) for k in txt_line[counter:(counter+6)]]
            line3d = _base.Line3d(infos[:3], infos[3:])
            counter += 6
            line3d_list.append(line3d)
        line3d = line3d_list[0]
        n_supports = int(txt_line[counter])
        counter += 1
        # collect supports
        img_id_list, line_id_list, line2d_list = [], [], []
        for supp_id in range(n_supports):
            img_id = int(txt_line[counter])
            counter += 1
            line_id = int(txt_line[counter])
            counter += 1
            infos = [float(k) for k in txt_line[counter:(counter+4)]]
            line2d = _base.Line2d(infos[:2], infos[2:])
            counter += 4
            img_id_list.append(img_id)
            line_id_list.append(line_id)
            line2d_list.append(line2d)
        track = _base.LineTrack(line3d, img_id_list, line_id_list, line2d_list)
        linetracks.append(track)
        for idx in range(n_lines):
            line_counts.append(track.count_images())
            line_track_id_list.append(len(linetracks) - 1)

    # construct matrix
    mergemat = np.zeros((len(linetracks), line_counters))
    for idx, track_id in enumerate(line_track_id_list):
        mergemat[track_id, idx] = 1
    return linetracks, line_track_id_list, line_counts, mergemat

def read_lines_from_input(input_file):
    '''
    General reader for lines
    '''
    if not os.path.exists(input_file):
        raise ValueError("Error! Input file/directory {0} not found.".format(input_file))

    # linetracks folder
    if not os.path.isfile(input_file):
        linetracks = read_folder_linetracks(input_file)
        lines = [track.line for track in linetracks]
        return lines, linetracks

    # npy file
    if input_file.endswith('.npy'):
        lines_np = read_npy(input_file)
        lines = [_base.Line3d(arr) for arr in lines_np.tolist()]
        return lines, None

    # obj file
    if input_file.endswith('.obj'):
        lines_np = load_obj(input_file)
        lines = [_base.Line3d(arr) for arr in lines_np.tolist()]
        return lines, None

    # line3dpp format
    if input_file.endswith('.txt'):
        linetracks, _, _, _ = read_txt_Line3Dpp(input_file)
        lines = [track.line for track in linetracks]
        return lines, linetracks

    # exception
    raise ValueError("Error! File {0} not supported. should be txt, obj, or folder to the linetracks.".format(input_dir))

def exists_txt_segments(folder, img_id):
    fname = os.path.join(folder, 'segments_{0}.txt'.format(img_id))
    return os.path.exists(fname)

def save_txt_segments(folder, img_id, segs):
    fname = os.path.join(folder, 'segments_{0}.txt'.format(img_id))
    n_segments = segs.shape[0]
    with open(fname, 'w') as f:
        f.write("{0}\n".format(n_segments))
        for line_id in range(n_segments):
            line = segs[line_id]
            f.write("{0} {1} {2} {3}\n".format(line[0], line[1], line[2], line[3]))

def read_txt_segments(folder, img_id):
    check_path(folder)
    fname = os.path.join(folder, 'segments_{0}.txt'.format(img_id))
    with open(fname, 'r') as f:
        txt_lines = f.readlines()
    n_segments = int(txt_lines[0].strip())
    assert n_segments + 1 == len(txt_lines)
    segs = []
    for idx in range(n_segments):
        k = txt_lines[idx + 1].strip().split(' ')
        seg = [float(kk) for kk in k]
        segs.append(seg)
    segs = np.array(segs)
    return segs

def read_all_segments_from_folder(folder):
    flist = os.listdir(folder)
    all_2d_segs = {}
    for fname in flist:
        img_id = int(fname[9:-4])
        segs = read_txt_segments(folder, img_id)
        all_2d_segs[img_id] = segs
    return all_2d_segs


