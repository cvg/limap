from pathlib import Path
import numpy as np
import torch
import PIL.Image
import pycolmap
import logging
from tqdm import tqdm

from hloc import localize_sfm
from hloc.utils.read_write_model import (read_model, write_model)
from hloc.pipelines.Cambridge.utils import create_query_list_with_intrinsics, evaluate
from hloc.utils.parsers import *

logger = logging.getLogger('hloc')


def create_reference_sfm(full_model, ref_model, blacklist=None, ext='.bin'):
    '''Create a new COLMAP model with only training images.'''
    logger.info('Creating the reference model.')
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if blacklist is not None:
        with open(blacklist, 'r') as f:
            blacklist = f.read().rstrip().split('\n')

    train_ids = []
    test_ids = []
    images_ref = dict()
    for id_, image in images.items():
        if blacklist and image.name in blacklist:
            test_ids.append(id_)
            continue
        train_ids.append(id_)
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, '.bin')
    logger.info(f'Kept {len(images_ref)} images out of {len(images)}.')
    return train_ids, test_ids

def scene_coordinates(p2D, R_w2c, t_w2c, depth, camera):
    assert len(depth) == len(p2D)
    ret = pycolmap.image_to_world(p2D, camera._asdict())
    p2D_norm = np.asarray(ret['world_points'])
    p2D_h = np.concatenate([p2D_norm, np.ones_like(p2D_norm[:, :1])], 1)
    p3D_c = p2D_h * depth[:, None]
    p3D_w = (p3D_c - t_w2c) @ R_w2c
    return p3D_w


def interpolate_depth(depth, kp):
    h, w = depth.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    depth = torch.from_numpy(depth)[None, None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        depth, kp, align_corners=True, mode='bilinear')[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        depth, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    interp_depth = interp.T.numpy().flatten()
    valid = valid.numpy()
    return interp_depth, valid


def image_path_to_rendered_depth_path(image_name):
    parts = image_name.split('/')
    name = '_'.join([''.join(parts[0].split('-')), parts[1]])
    name = name.replace('color', 'pose')
    name = name.replace('png', 'depth.tiff')
    return name


def project_to_image(p3D, R, t, camera, eps: float = 1e-4, pad: int = 1):
    p3D = (p3D @ R.T) + t
    visible = p3D[:, -1] >= eps  # keep points in front of the camera
    p2D_norm = p3D[:, :-1] / p3D[:, -1:].clip(min=eps)
    ret = pycolmap.world_to_image(p2D_norm, camera._asdict())
    p2D = np.asarray(ret['image_points'])
    size = np.array([camera.width - pad - 1, camera.height - pad - 1])
    valid = np.all((p2D >= pad) & (p2D <= size), -1)
    valid &= visible
    return p2D[valid], valid


def correct_sfm_with_gt_depth(sfm_path, depth_folder_path, output_path):
    cameras, images, points3D = read_model(sfm_path)
    logger.info("Correcting sfm using depth...")
    for imgid, img in tqdm(images.items()):
        image_name = img.name
        depth_name = image_path_to_rendered_depth_path(image_name)

        depth = PIL.Image.open(Path(depth_folder_path) / depth_name)
        depth = np.array(depth).astype('float64')
        depth = depth/1000.  # mm to meter
        depth[(depth == 0.0) | (depth > 1000.0)] = np.nan

        R_w2c, t_w2c = img.qvec2rotmat(), img.tvec
        camera = cameras[img.camera_id]
        p3D_ids = img.point3D_ids
        p3Ds = np.stack([points3D[i].xyz for i in p3D_ids[p3D_ids != -1]], 0)

        p2Ds, valids_projected = project_to_image(p3Ds, R_w2c, t_w2c, camera)
        invalid_p3D_ids = p3D_ids[p3D_ids != -1][~valids_projected]
        interp_depth, valids_backprojected = interpolate_depth(depth, p2Ds)
        scs = scene_coordinates(p2Ds[valids_backprojected], R_w2c, t_w2c,
                                interp_depth[valids_backprojected],
                                camera)
        invalid_p3D_ids = np.append(
            invalid_p3D_ids,
            p3D_ids[p3D_ids != -1][valids_projected][~valids_backprojected])
        for p3did in invalid_p3D_ids:
            if p3did == -1:
                continue
            else:
                obs_imgids = points3D[p3did].image_ids
                invalid_imgids = list(np.where(obs_imgids == img.id)[0])
                points3D[p3did] = points3D[p3did]._replace(
                    image_ids=np.delete(obs_imgids, invalid_imgids),
                    point2D_idxs=np.delete(points3D[p3did].point2D_idxs,
                                           invalid_imgids))

        new_p3D_ids = p3D_ids.copy()
        sub_p3D_ids = new_p3D_ids[new_p3D_ids != -1]
        valids = np.ones(np.count_nonzero(new_p3D_ids != -1), dtype=bool)
        valids[~valids_projected] = False
        valids[valids_projected] = valids_backprojected
        sub_p3D_ids[~valids] = -1
        new_p3D_ids[new_p3D_ids != -1] = sub_p3D_ids
        img = img._replace(point3D_ids=new_p3D_ids)

        assert len(img.point3D_ids[img.point3D_ids != -1]) == len(scs), (
                f"{len(scs)}, {len(img.point3D_ids[img.point3D_ids != -1])}")
        for i, p3did in enumerate(img.point3D_ids[img.point3D_ids != -1]):
            points3D[p3did] = points3D[p3did]._replace(xyz=scs[i])
        images[imgid] = img

    output_path.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, output_path)

