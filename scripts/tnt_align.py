import os

import numpy as np

MAX_ERROR = 0.01
colmap_output_path = os.path.expanduser("~/data/TanksTemples/colmap/training")
input_meta_path = os.path.expanduser("~/data/TanksTemples/meta_train")


def get_imname_list(scene_id):
    image_path = os.path.join(colmap_output_path, scene_id, "dense/images")
    flist = os.listdir(image_path)
    n_images = len(flist)
    imname_list = []
    for idx in range(n_images):
        fname = f"{idx + 1:06d}.jpg"
        # fname = os.path.join(image_path, fname)
        imname_list.append(fname)
    return imname_list


def read_positions(log_file):
    with open(log_file) as f:
        lines = f.readlines()
    n_images = int(len(lines) / 5)
    positions = []
    counter = 0
    for _ in range(n_images):
        counter += 1
        mat = []
        for _ in range(4):
            line = lines[counter].strip("\n").split(" ")
            mat.append([float(k) for k in line])
            counter += 1
        mat = np.array(mat)
        pos = mat[:3, 3]
        positions.append(pos)
    return positions


def read_trans(fname):
    with open(fname) as f:
        lines = f.readlines()
    mat = []
    for idx in range(4):
        line = lines[idx].strip("\n").split(" ")
        mat.append([float(k) for k in line])
    mat = np.array(mat)
    assert np.all(mat[3, :] == np.array([0, 0, 0, 1]))
    return mat[:3, :]


def write_geoinfo_txt(fname, imname_list, positions):
    with open(fname, "w") as f:
        for imname, pos in zip(imname_list, positions):
            f.write(f"{imname} {pos[0]} {pos[1]} {pos[2]}\n")


def main():
    scene_id_list = os.listdir(input_meta_path)
    for scene_id in scene_id_list:
        # get geo txt
        imname_list = get_imname_list(scene_id)
        log_file = os.path.join(
            input_meta_path, scene_id, f"{scene_id}_COLMAP_SfM.log"
        )
        positions = read_positions(log_file)
        trans_file = os.path.join(
            input_meta_path, scene_id, f"{scene_id}_trans.txt"
        )
        trans_mat = read_trans(trans_file)
        new_positions = [
            trans_mat[:3, :3] @ k + trans_mat[:3, 3] for k in positions
        ]
        output_fname = os.path.join(
            input_meta_path, scene_id, "geo_positions.txt"
        )
        write_geoinfo_txt(output_fname, imname_list, new_positions)

        # colmap align
        cmd_list = []
        basepath = os.path.join(colmap_output_path, scene_id, "dense")
        cmd = "mkdir -p {}".format(os.path.join(basepath, "aligned"))
        cmd_list.append(cmd)
        cmd = "colmap model_aligner --input_path {} --output_path {} \
               --ref_images_path {} \
               --robust_alignment 1 \
               --robust_alignment_max_error {} \
               --transform_path {} \
               --ref_is_gps false".format(
            os.path.join(basepath, "sparse"),
            os.path.join(basepath, "aligned"),
            os.path.join(input_meta_path, scene_id, "geo_positions.txt"),
            MAX_ERROR,
            os.path.join(basepath, "transform.txt"),
        )
        cmd_list.append(cmd)
        cmd = "colmap model_converter --input_path {} --output_path {} \
               --output_type PLY".format(
            os.path.join(basepath, "aligned"),
            os.path.join(basepath, "aligned/points.ply"),
        )
        cmd_list.append(cmd)
        for cmd in cmd_list:
            print(cmd)
            os.system(cmd)


if __name__ == "__main__":
    main()
