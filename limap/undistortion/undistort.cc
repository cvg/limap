#include "undistortion/undistort.h"

#include <iostream>
#include <colmap/util/bitmap.h>
#include <colmap/base/camera.h>
#include <colmap/base/undistortion.h>

namespace limap {

namespace undistortion {

Camera UndistortCamera(const std::string& imname_in, const Camera& camera, const std::string& imname_out) {
    colmap::Bitmap img, img_undistorted;
    img.Read(imname_in);
    colmap::Camera cam = camera;

    bool exif_autorotate = false;
    if (cam.Height() != img.Height() || cam.Width() != img.Width()) {
        if (cam.Width() != img.Height() || cam.Height() != img.Width())
            throw std::runtime_error("Error! The height and width of the given camera do not match the input image.");
        // std::cout<<"[WARNING] Auto rotating image (EXIF): "<<imname_in<<std::endl;
        exif_autorotate = true;
        cam.Rescale(img.Width(), img.Height());
    }
    colmap::UndistortCameraOptions undist_options;
    colmap::Camera cam_undistorted;
    colmap::UndistortImage(undist_options, img, cam, &img_undistorted, &cam_undistorted);
    img_undistorted.Write(imname_out);
    if (exif_autorotate)
        cam_undistorted.Rescale(img_undistorted.Height(), img_undistorted.Width());

    Camera camera_undistorted = cam_undistorted;
    return camera_undistorted;
}

CameraView UndistortCameraView(const std::string& imname_in, const CameraView& view, const std::string& imname_out) {
    CameraView new_view = view;
    new_view.cam = UndistortCamera(imname_in, view.cam, imname_out);
    return new_view;
}

V2D UndistortPoint(const V2D& point, const Camera& distorted_camera, const Camera& undistorted_camera) {
    return undistorted_camera.WorldToImage(distorted_camera.ImageToWorld(point));
}

std::vector<V2D> UndistortPoints(const std::vector<V2D>& points, const Camera& distorted_camera, const Camera& undistorted_camera) {
    std::vector<V2D> new_points;
    for (auto it = points.begin(); it != points.end(); ++it) {
        new_points.push_back(UndistortPoint(*it, distorted_camera, undistorted_camera));
    }
    return new_points;
}

} // namespace undistortion

} // namespace limap

