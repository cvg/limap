import cv2
from _limap import _base, _undistortion


def undistort_image_camera(camera, imname_in, imname_out):
    """
    Run COLMAP undistortion on one single image with an input camera. \
    The undistortion is only applied if the camera model is \
    neither "SIMPLE_PINHOLE" nor "PINHOLE".

    Args:
        camera (:class:`limap.base.Camera`): \
            The camera (type + parameters) for the image.
        imname_in (str): filename for the input image
        imname_out (str): filename for the output undistorted image

    Returns:
        :class:`limap.base.Camera`: The undistorted camera
    """
    if camera.IsUndistorted():  # no distortion
        img = cv2.imread(imname_in)
        cv2.imwrite(imname_out, img)
        if int(camera.model) == 0 or int(camera.model) == 1:
            return camera
        # if "SIMPLE_RADIAL", update to "SIMPLE_PINHOLE"
        if int(camera.model) == 2:
            new_camera = _base.Camera(
                "SIMPLE_PINHOLE",
                camera.K(),
                cam_id=camera.camera_id,
                hw=[camera.h(), camera.w()],
            )
        else:
            # else change to pinhole
            new_camera = _base.Camera(
                "PINHOLE",
                camera.K(),
                cam_id=camera.camera_id,
                hw=[camera.h(), camera.w()],
            )
        return new_camera

    # undistort
    camera_undistorted = _undistortion._UndistortCamera(
        imname_in, camera, imname_out
    )
    return camera_undistorted


def undistort_points(points, distorted_camera, undistorted_camera):
    """
    Run COLMAP undistortion on the keypoints.

    Args:
        points (list[:class:`np.array`]): \
            List of 2D keypoints on the distorted image
        distorted_camera (:class:`limap.base.Camera`): \
            The camera before undistortion
        undistorted_camera (:class:`limap.base.Camera`): \
            The camera after undistortion

    Returns:
        list[:class:`np.array`]: \
            List of the corresponding 2D keypoints on the undistorted image
    """
    return _undistortion._UndistortPoints(
        points, distorted_camera, undistorted_camera
    )
