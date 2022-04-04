#ifndef LIMAP_BASE_CAMERA_VIEW_H
#define LIMAP_BASE_CAMERA_VIEW_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <cmath>
#include <fstream>

namespace py = pybind11;

#include "util/types.h"
#include "_limap/helpers.h"

#include "base/camera.h"

namespace limap {

class CameraView {
public:
    CameraView() {}
    CameraView(const Camera& input_cam, const CameraPose& input_pose, const std::string& image_name = "none"): cam(input_cam), pose(input_pose), image_name_(image_name) {}
    CameraView(py::dict dict);

    Camera cam;
    CameraPose pose;
    
    py::dict as_dict() const;
    M3D K() const { return cam.K(); }
    M3D K_inv() const { return cam.K_inv(); }
    int h() const { return cam.h(); }
    int w() const { return cam.w(); }
    M3D R() const { return pose.R(); }
    V3D T() const { return pose.T(); }

    V2D projection(const V3D& p3d) const;
    V3D ray_direction(const V2D& p2d) const;

    // image
    void SetImageName(const std::string& image_name) { image_name_ = image_name; }
    std::string image_name() const { return image_name_; }

private:
    std::string image_name_;
};

// used for optimization
class MinimalPinholeCamera {
public:
    MinimalPinholeCamera() {}
    MinimalPinholeCamera(const CameraView& view);
    CameraView GetCameraView() const;

    V4D kvec; // [f1, f2, c1, c2]
    V4D qvec;
    V3D tvec;
    int height, width;
};

// interchanging between CameraView and MinimalPinholeCamera
MinimalPinholeCamera cam2minimalcam(const CameraView& view);

CameraView minimalcam2cam(const MinimalPinholeCamera& camera);

} // namespace limap

#endif

