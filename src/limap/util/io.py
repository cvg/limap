import os
import shutil

import numpy as np
from pycolmap import logging


def check_directory(fname):
    if fname is None:
        raise ValueError("Error! Input directory is None!")
    if os.path.dirname(fname) != "" and (
        not os.path.exists(os.path.dirname(fname))
    ):
        raise ValueError(
            f"Error! Base directory {os.path.dirname(fname)} does not exist!"
        )


def check_path(fname):
    if fname is None:
        raise ValueError("Error! Input filepath is None!")
    if not os.path.exists(fname):
        raise ValueError(f"Error! File {fname} does not exist!")


def delete_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)


def check_makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_npy(fname, nparray):
    check_directory(fname)
    with open(fname, "wb") as f:
        np.save(f, np.array(nparray, dtype=object))


def read_npy(fname):
    check_path(fname)
    with open(fname, "rb") as f:
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
    points = [
        (points[i, 0], points[i, 1], points[i, 2])
        for i in range(points.shape[0])
    ]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    PlyData([el], text=True).write(fname)


def read_ply(fname):
    from plyfile import PlyData

    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data["x"])
    y = np.asarray(plydata.elements[0].data["y"])
    z = np.asarray(plydata.elements[0].data["z"])
    points = np.stack([x, y, z], axis=1)
    logging.info(f"number of points: {points.shape[0]}")
    return points


def save_txt_metainfos(fname, neighbors, ranges):
    """
    Write out .txt for neighbors and ranges
    """
    check_directory(fname)
    with open(fname, "w") as f:
        f.write(f"number of images, {len(neighbors)}\n")
        f.write(f"x-range, {ranges[0][0]}, {ranges[1][0]}\n")
        f.write(f"y-range, {ranges[0][1]}, {ranges[1][1]}\n")
        f.write(f"z-range, {ranges[0][2]}, {ranges[1][2]}\n")
        for img_id, neighbor in neighbors.items():
            str_ = f"image {img_id}"
            for ng_idx in neighbor:
                str_ += f", {ng_idx}"
            f.write(str_ + "\n")


def read_txt_metainfos(fname):
    """
    Read in .txt for neighbors and ranges
    """
    check_path(fname)
    with open(fname) as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(",")[1])
    counter += 1
    ranges = (np.zeros(3), np.zeros(3))
    k = txt_lines[counter].strip().split(",")[1:]
    ranges[0][0], ranges[1][0] = float(k[0]), float(k[1])
    counter += 1
    k = txt_lines[counter].strip().split(",")[1:]
    ranges[0][1], ranges[1][1] = float(k[0]), float(k[1])
    counter += 1
    k = txt_lines[counter].strip().split(",")[1:]
    ranges[0][2], ranges[1][2] = float(k[0]), float(k[1])
    counter += 1
    neighbors = {}
    for _ in range(n_images):
        k = txt_lines[counter].strip().split(",")
        img_id = int(k[0][6:])
        neighbor = [int(kk) for kk in k[1:]]
        neighbors[img_id] = neighbor
        counter += 1
    return neighbors, ranges


def save_txt_imname_list(fname, imname_list):
    check_directory(fname)
    with open(fname, "w") as f:
        f.write(f"number of images, {len(imname_list)}\n")
        for imname in imname_list:
            f.write(imname + "\n")


def read_txt_imname_list(fname):
    check_path(fname)
    with open(fname) as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(",")[1])
    counter += 1
    imname_list = []
    for _ in range(n_images):
        imname = txt_lines[counter].strip()
        imname_list.append(imname)
        counter += 1
    return imname_list


def save_txt_imname_dict(fname, imname_dict):
    check_directory(fname)
    with open(fname, "w") as f:
        f.write(f"number of images, {len(imname_dict)}\n")
        for img_id, imname in imname_dict.items():
            f.write(f"{img_id}, {imname}\n")


def read_txt_imname_dict(fname):
    check_path(fname)
    with open(fname) as f:
        txt_lines = f.readlines()
    counter = 0
    n_images = int(txt_lines[counter].strip().split(",")[1])
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
    if isinstance(lines, list):
        if isinstance(lines[0], np.ndarray):
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
    with open(fname, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for idx in range(n_lines):
            f.write(f"l {2 * idx + 1} {2 * idx + 2}\n")


def load_obj(fname):
    with open(fname) as f:
        flines = f.readlines()
    counter = 0
    vertices = []
    while True:
        fline = flines[counter].strip("\n").split(" ")
        if fline[0] != "v":
            break
        vertice = np.array([float(fline[1]), float(fline[2]), float(fline[3])])
        vertices.append(vertice)
        counter += 1
    vertices = np.array(vertices)
    n_lines = int(vertices.shape[0] / 2)
    lines = vertices.reshape((n_lines, 2, 3))
    return lines


def exists_txt_segments(folder, img_id):
    fname = os.path.join(folder, f"segments_{img_id}.txt")
    return os.path.exists(fname)


def save_txt_segments(folder, img_id, segs):
    fname = os.path.join(folder, f"segments_{img_id}.txt")
    n_segments = segs.shape[0]
    with open(fname, "w") as f:
        f.write(f"{n_segments}\n")
        for line_id in range(n_segments):
            line = segs[line_id]
            f.write(f"{line[0]} {line[1]} {line[2]} {line[3]}\n")


def read_txt_segments(folder, img_id):
    check_path(folder)
    fname = os.path.join(folder, f"segments_{img_id}.txt")
    with open(fname) as f:
        txt_lines = f.readlines()
    n_segments = int(txt_lines[0].strip())
    assert n_segments + 1 == len(txt_lines)
    segs = []
    for idx in range(n_segments):
        k = txt_lines[idx + 1].strip().split(" ")
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
