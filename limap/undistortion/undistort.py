from _limap import _base, _undistortion
import os, sys
import cv2
import numpy as np
import copy

def UndistortImageParams(K, dist_coeffs, img_hw, imname_in, imname_out):
    camera = _base.Camera(K, np.eye(3), np.zeros((3)), dist_coeffs)
    camera.set_hw(img_hw[0], img_hw[1])
    camera_undistorted = UndistortImageCamera(camera, imname_in, imname_out)
    return camera_undistorted.K, (camera_undistorted.h, camera_undistorted.w)

def UndistortImageCamera(camera, imname_in, imname_out):
    if camera.checkUndistorted(): # no distortion
        img = cv2.imread(imname_in)
        cv2.imwrite(imname_out, img)
        return camera
    camera_undistorted = _undistortion._Undistort(imname_in, camera, imname_out)
    return camera_undistorted

def UndistortImageCamera_OPENCV(camera, imname_in, imname_out):
    img = cv2.imread(imname_in)
    imsize = (img.shape[1], img.shape[0])
    K = camera.K
    coeffs = np.array(camera.dist_coeffs)
    map1, map2 = cv2.initUndistortRectifyMap(K, coeffs, np.eye(3), K, imsize, cv2.CV_16SC2)
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cv2.imwrite(imname_out, img_undistorted)
    # deepcopy
    camera_undistorted = _base.Camera(camera.K, camera.R, camera.T)
    camera_undistorted.set_hw(camera.h, camera.w)
    return camera_undistorted


