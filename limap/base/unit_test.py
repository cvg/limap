import _limap._base as _base
import numpy as np
from pycolmap import logging


def unit_test_add_noise(imagecols):
    dict_imagecols = imagecols.as_dict()
    # # perturb cameras
    # m_cameras = dict_imagecols["cameras"]
    # for cam_id in imagecols.get_cam_ids():
    #     n_params = len(m_cameras[cam_id]["params"])
    #     m_cameras[cam_id]["params"] += np.random.normal(0, 10, n_params);
    # perturb poses
    m_images = dict_imagecols["images"]
    for img_id in imagecols.get_img_ids():
        m_images[img_id]["qvec"] += np.random.normal(0, 0.001, 4)
        m_images[img_id]["tvec"] += np.random.normal(0, 0.01, 3)
    return _base.ImageCollection(dict_imagecols)


def report_error(imagecols_pred, imagecols):
    # cameras
    camera_errors = []
    for cam_id in imagecols.get_cam_ids():
        error = np.array(imagecols_pred.cam(cam_id).params) - np.array(
            imagecols.cam(cam_id).params
        )
        error = np.abs(error)
        camera_errors.append(error)
    logging.info("camera_errors", np.array(camera_errors).mean(0))

    # images
    pose_errors = []
    for img_id in imagecols.get_img_ids():
        R_error = (
            imagecols_pred.camimage(img_id).R() - imagecols.camimage(img_id).R()
        )
        R_error = np.sqrt(np.sum(R_error**2))
        T_error = (
            imagecols_pred.camimage(img_id).T() - imagecols.camimage(img_id).T()
        )
        T_error = np.sqrt(np.sum(T_error**2))
        pose_errors.append(np.array([R_error, T_error]))
    logging.info("pose_error: (R, T)", np.array(pose_errors).mean(0))
