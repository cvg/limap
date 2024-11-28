import cv2
import numpy as np
from pycolmap import logging

import limap.base as base


def compute_rot_err(R1, R2):
    rot_err = R1[0:3, 0:3].T.dot(R2[0:3, 0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1, 3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.0
    return rot_err[0]


def compute_pose_err(pose, pose_gt):
    """
    Inputs:
    - pose:     base.CameraPose
    - pose_gt:  base.CameraPose
    """
    trans_err = np.linalg.norm(pose.center() - pose_gt.center())
    rot_err = compute_rot_err(pose.R(), pose_gt.R())
    return trans_err, rot_err


def eval_imagecols(
    imagecols, imagecols_gt, max_error=0.01, enable_logging=True
):
    if enable_logging:
        logging.info(
            f"[LOG EVAL] imagecols.NumImages() = {imagecols.NumImages()}"
        )
        logging.info(
            f"[LOG EVAL] imagecols_gt.NumImages() = {imagecols_gt.NumImages()}"
        )
    _, imagecols_aligned = base.align_imagecols(
        imagecols,
        imagecols_gt.subset_by_image_ids(imagecols.get_img_ids()),
        max_error=max_error,
    )
    shared_img_ids = list(
        set(imagecols.get_img_ids()) & set(imagecols_gt.get_img_ids())
    )
    assert len(shared_img_ids) == imagecols.NumImages()
    imagecols_gt = imagecols_gt.subset_by_image_ids(shared_img_ids)
    trans_errs, rot_errs = [], []
    for img_id in shared_img_ids:
        pose = imagecols_aligned.camimage(img_id).pose
        pose_gt = imagecols_gt.camimage(img_id).pose
        trans_err, rot_err = compute_pose_err(pose, pose_gt)
        trans_errs.append(trans_err)
        rot_errs.append(rot_err)
    return trans_errs, rot_errs


def eval_imagecols_relpose(
    imagecols, imagecols_gt, fill_uninitialized=True, enable_logging=True
):
    shared_img_ids = list(
        set(imagecols.get_img_ids()) & set(imagecols_gt.get_img_ids())
    )
    assert len(shared_img_ids) == imagecols.NumImages()
    if enable_logging:
        logging.info(
            f"[LOG EVAL] imagecols.NumImages() = {imagecols.NumImages()}"
        )
        logging.info(
            f"[LOG EVAL] imagecols_gt.NumImages() = {imagecols_gt.NumImages()}"
        )
    if fill_uninitialized:
        img_ids = imagecols_gt.get_img_ids()
    else:
        img_ids = shared_img_ids
    num_images = len(img_ids)
    err_list = []
    for i in range(num_images - 1):
        if imagecols.exist_image(img_ids[i]):
            pose1 = imagecols.camimage(img_ids[i]).pose
        else:
            pose1 = base.CameraPose()
        pose1_gt = imagecols_gt.camimage(img_ids[i]).pose
        for j in range(i + 1, num_images):
            if imagecols.exist_image(img_ids[j]):
                pose2 = imagecols.camimage(img_ids[j]).pose
            else:
                pose2 = base.CameraPose()
            pose2_gt = imagecols_gt.camimage(img_ids[j]).pose

            relR = pose1.R() @ pose2.R().T
            relT = pose1.T() - relR @ pose2.T()
            relT_vec = relT / np.linalg.norm(relT)
            relR_gt = pose1_gt.R() @ pose2_gt.R().T
            relT_gt = pose1_gt.T() - relR_gt @ pose2_gt.T()
            relT_gt_vec = relT_gt / np.linalg.norm(relT_gt)

            rot_err = compute_rot_err(relR, relR_gt)
            t_angle = (
                np.arccos(np.abs(relT_vec.dot(relT_gt_vec))) * 180.0 / np.pi
            )
            err = max(rot_err, t_angle)
            err_list.append(err)
    return np.array(err_list)
