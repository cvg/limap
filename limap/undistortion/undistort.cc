#include "undistortion/undistort.h"

#include <colmap/util/bitmap.h>
#include <colmap/base/camera.h>
#include <colmap/base/undistortion.h>

namespace limap {

namespace undistortion {

Camera Undistort(const std::string& imname_in, const Camera& camera, const std::string& imname_out) {
    colmap::Bitmap img, img_undistorted;
    img.Read(imname_in);
    colmap::Camera cam = camera;
    colmap::UndistortCameraOptions undist_options;

    colmap::Camera cam_undistorted;
    colmap::UndistortImage(undist_options, img, cam, &img_undistorted, &cam_undistorted);
    img_undistorted.Write(imname_out);

    Camera camera_undistorted = cam_undistorted;
    return camera_undistorted;
}

CameraView Undistort(const std::string& imname_in, const CameraView& view, const std::string& imname_out) {
    CameraView new_view = view;
    new_view.cam = Undistort(imname_in, view.cam, imname_out);
    return new_view;
}

} // namespace undistortion

} // namespace limap

