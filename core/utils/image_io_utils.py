import numpy as np
import cv2
import h5py

def read_image(imname, resize_hw=None, max_image_dim=None, set_gray=True):
    img = cv2.imread(imname)
    if resize_hw is not None:
        img = cv2.resize(img, (resize_hw[1], resize_hw[0]))
    if (max_image_dim is not None) and max_image_dim != -1:
        hw_now = img.shape[:2]
        ratio = max_image_dim / max(hw_now[0], hw_now[1])
        if ratio < 1.0:
            h_new = int(round(hw_now[0] * ratio))
            w_new = int(round(hw_now[1] * ratio))
            img = cv2.resize(img, (w_new, h_new))
    if set_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def read_raydepth(raydepth_fname, resize_hw=None, max_image_dim=None):
    with h5py.File(raydepth_fname, 'r') as f:
        raydepth = np.array(f['dataset']).astype(np.float32)
    if resize_hw is not None and raydepth.shape != resize_hw:
        raydepth = cv2.resize(raydepth, (resize_hw[1], resize_hw[0]))
    if (max_image_dim is not None) and max_image_dim != -1:
        hw_now = raydepth.shape[:2]
        ratio = max_image_dim / max(hw_now[0], hw_now[1])
        if ratio < 1.0:
            h_new = int(round(hw_now[0] * ratio))
            w_new = int(round(hw_now[1] * ratio))
            raydepth = cv2.resize(raydepth, (w_new, h_new))
    return raydepth

def raydepth2depth(raydepth, K, img_hw):
    K_inv = np.linalg.inv(K)
    h, w = raydepth.shape[0], raydepth.shape[1]
    grids = np.meshgrid(np.arange(w), np.arange(h))
    coords_homo = [grids[0].reshape(-1), grids[1].reshape(-1), np.ones((h*w))]
    coords_homo = np.stack(coords_homo)
    coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)
    coeffs = coeffs.reshape(h, w)
    depth = raydepth / coeffs
    return depth


