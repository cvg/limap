import _limap._base as _base
import numpy as np


def umeyama_alignment(x, y, with_scale=True):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise AssertionError("x.shape not equal to y.shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))
    return r, t, c


def align_imagecols_umeyama(imagecols_src, imagecols_dst):
    # assertion check
    assert imagecols_src.NumImages() == imagecols_dst.NumImages()
    assert np.all(imagecols_src.get_img_ids() == imagecols_dst.get_img_ids())

    # fit transformation
    xyz_src = np.array(imagecols_src.get_locations()).transpose()
    xyz_dst = np.array(imagecols_dst.get_locations()).transpose()
    r, t, c = umeyama_alignment(xyz_src, xyz_dst, with_scale=True)
    transform = _base.SimilarityTransform3(r, t, c)
    imagecols_aligned = imagecols_src.apply_similarity_transform(transform)
    return transform, imagecols_aligned


def align_imagecols_colmap(
    imagecols_src,
    imagecols_dst,
    max_error=0.01,
    tmp_folder="tmp/model_convertion",
):
    import os
    import subprocess

    import numpy as np

    import limap.util.io as limapio
    from limap.pointsfm.model_converter import convert_imagecols_to_colmap

    # assertion check
    assert imagecols_src.NumImages() == imagecols_dst.NumImages()
    assert np.all(imagecols_src.get_img_ids() == imagecols_dst.get_img_ids())

    limapio.check_makedirs(tmp_folder)
    src_folder = os.path.join(tmp_folder, "source")
    tgt_folder = os.path.join(tmp_folder, "target")
    limapio.check_makedirs(src_folder)
    limapio.check_makedirs(tgt_folder)

    # write source imagecols into COLMAP format
    convert_imagecols_to_colmap(imagecols_src, src_folder)

    # write the positions of target imagecols
    fname_positions = os.path.join(tmp_folder, "geo_positions.txt")
    with open(fname_positions, "w") as f:
        for img_id in imagecols_src.get_img_ids():
            imname = imagecols_src.image_name(img_id)
            pos = imagecols_dst.camview(img_id).pose.center()
            f.write(f"{imname} {pos[0]} {pos[1]} {pos[2]}\n")

    # call comlap model aligner
    # TODO: use pycolmap
    transform_path = os.path.join(tmp_folder, "transform.txt")
    cmd = [
        "colmap",
        "model_aligner",
        "--input_path",
        src_folder,
        "--output_path",
        tgt_folder,
        "--ref_images_path",
        fname_positions,
        "--robust_alignment",
        str(1),
        "--robust_alignment_max_error",
        str(max_error),
        "--transform_path",
        transform_path,
        "--ref_is_gps",
        "false",
    ]
    subprocess.run(cmd)
    if not os.path.exists(transform_path):
        return None, None

    # read in transformation
    def read_trans(fname):
        with open(fname) as f:
            lines = f.readlines()
        mat = []
        for idx in range(4):
            line = lines[idx].strip("\n").split()
            mat.append([float(k) for k in line])
        mat = np.array(mat)
        assert np.all(mat[3, :] == np.array([0, 0, 0, 1]))
        return mat[:3, :]

    transform = read_trans(transform_path)

    scale = np.linalg.norm(transform[:, 0])
    R = transform[:3, :3] / scale
    t = transform[:3, 3]
    transform = _base.SimilarityTransform3(R, t, scale)
    imagecols_aligned = imagecols_src.apply_similarity_transform(transform)

    # delete tmp folder
    limapio.delete_folder(tmp_folder)
    return transform, imagecols_aligned


def align_imagecols(imagecols_src, imagecols_dst, max_error=0.01):
    return align_imagecols_colmap(
        imagecols_src, imagecols_dst, max_error=max_error
    )
