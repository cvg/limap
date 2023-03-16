import numpy as np
import cv2
import limap.base as _base

def compute_rot_err(R1, R2):
    rot_err = R1[0:3,0:3].T.dot(R2[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def compute_pose_err(pose, pose_gt):
    '''
    Inputs:
    - pose:     _base.CameraPose
    - pose_gt:  _base.CameraPose
    '''
    trans_err = np.linalg.norm(pose.center() - pose_gt.center())
    rot_err = compute_rot_err(pose.R(), pose_gt.R())
    return trans_err, rot_err

def eval_imagecols(imagecols, imagecols_gt):
    _, imagecols_aligned = _base.align_imagecols(imagecols, imagecols_gt)
    shared_img_ids = list(set(imagecols.get_img_ids()) & set(imagecols_gt.get_img_ids()))
    assert len(shared_img_ids) == imagecols.NumImages();
    imagecols_gt = imagecols_gt.subset_by_image_ids(shared_img_ids)
    trans_errs, rot_errs = [], []
    for img_id in shared_img_ids:
        pose = imagecols_aligned.camimage(img_id).pose
        pose_gt = imagecols_gt.camimage(img_id).pose
        trans_err, rot_err = compute_pose_err(pose, pose_gt)
        trans_errs.append(trans_err)
        rot_errs.append(rot_err)
    return trans_errs, rot_errs

def eval_imagecols_relpose(imagecols, imagecols_gt):
    shared_img_ids = list(set(imagecols.get_img_ids()) & set(imagecols_gt.get_img_ids()))
    assert len(shared_img_ids) == imagecols.NumImages();

    num_shared = len(shared_img_ids)
    err_list = []
    for i in range(num_shared - 1):
        pose1 = imagecols.camimage(shared_img_ids[i]).pose
        pose1_gt = imagecols_gt.camimage(shared_img_ids[i]).pose
        for j in range(i + 1, num_shared):
            pose2 = imagecols.camimage(shared_img_ids[j]).pose
            pose2_gt = imagecols_gt.camimage(shared_img_ids[j]).pose

            relR = pose1.R() @ pose2.R().T
            relT = pose1.T() - relR @ pose2.T()
            relT_vec = relT / np.linalg.norm(relT)
            relR_gt = pose1_gt.R() @ pose2_gt.R().T
            relT_gt = pose1_gt.T() - relR_gt @ pose2_gt.T()
            relT_gt_vec = relT_gt / np.linalg.norm(relT_gt)

            rot_err = compute_rot_err(relR, relR_gt)
            t_angle = np.arccos(np.abs(relT_vec.dot(relT_gt_vec))) * 180.0 / np.pi
            err = max(rot_err, t_angle)
            err_list.append(err)
    return np.array(err_list)


