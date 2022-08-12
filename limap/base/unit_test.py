import _limap._base as _base
import numpy as np

def unit_test_add_noise(imagecols):
    dict_imagecols = imagecols.as_dict()
    # perturb cameras
    m_cameras = dict_imagecols["cameras"]
    for cam_id in imagecols.get_cam_ids():
        n_params = len(m_cameras[cam_id]["params"])
        m_cameras[cam_id]["params"] += np.random.normal(0, 10, n_params);
    # perturb poses
    m_images = dict_imagecols["images"]
    for img_id in imagecols.get_img_ids():
        m_images[img_id]["qvec"] += np.random.normal(0, 0.01, 4)
        m_images[img_id]["tvec"] += np.random.normal(0, 0.1, 3)
    return _base.ImageCollection(dict_imagecols)

