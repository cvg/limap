from _limap import _base, _undistortion
import os, sys
import cv2
import numpy as np
import copy

def UndistortImageCamera(camera, imname_in, imname_out):
    if camera.IsUndistorted(): # no distortion
        img = cv2.imread(imname_in)
        cv2.imwrite(imname_out, img)
        if camera.model_id() == 0 or camera.model_id() == 1:
            return camera
        # else change to pinhole
        new_camera = _base.Camera("PINHOLE", camera.K(), cam_id = camera.cam_id(), hw=[camera.h(), camera.w()])
        return new_camera

    # undistort
    camera_undistorted = _undistortion._UndistortCamera(imname_in, camera, imname_out)
    return camera_undistorted

