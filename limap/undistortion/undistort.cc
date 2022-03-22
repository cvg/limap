#include "undistortion/undistort.h"

#include <colmap/util/bitmap.h>
#include <colmap/base/camera.h>
#include <colmap/base/undistortion.h>

namespace limap {

namespace undistortion {

PinholeCamera Undistort(const std::string& imname_in, const PinholeCamera& camera, const std::string& imname_out) {
    colmap::Bitmap img, img_undistorted;
    img.Read(imname_in);
    colmap::Camera cam, cam_undistorted;
    cam = cam_ours2colmap(camera);
    colmap::UndistortCameraOptions undist_options;

    colmap::UndistortImage(undist_options, img, cam, &img_undistorted, &cam_undistorted);
    img_undistorted.Write(imname_out);

    PinholeCamera camera_undistorted = cam_colmap2ours(cam_undistorted);
    camera_undistorted.R = camera.R;
    camera_undistorted.T = camera.T;
    return camera_undistorted;
}

} // namespace undistortion

} // namespace limap

