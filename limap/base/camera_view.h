#ifndef LIMAP_BASE_CAMERA_VIEW_H_
#define LIMAP_BASE_CAMERA_VIEW_H_

#include <cmath>
#include <fstream>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "_limap/helpers.h"
#include "util/types.h"

#include "base/camera.h"

namespace limap {

class CameraImage {
public:
  CameraImage() {}
  CameraImage(const int &input_cam_id, const std::string &image_name = "none")
      : cam_id(input_cam_id), pose(CameraPose(false)), image_name_(image_name) {
  } // empty image
  CameraImage(const Camera &input_cam, const std::string &image_name = "none")
      : cam_id(input_cam.camera_id), pose(CameraPose(false)),
        image_name_(image_name) {} // empty image
  CameraImage(const int &input_cam_id, const CameraPose &input_pose,
              const std::string &image_name = "none")
      : cam_id(input_cam_id), pose(input_pose), image_name_(image_name) {}
  CameraImage(const Camera &input_cam, const CameraPose &input_pose,
              const std::string &image_name = "none")
      : cam_id(input_cam.camera_id), pose(input_pose), image_name_(image_name) {
  }
  CameraImage(py::dict dict);
  CameraImage(const CameraImage &camimage)
      : cam_id(camimage.cam_id), pose(camimage.pose) {
    SetImageName(camimage.image_name());
  }

  int cam_id;
  CameraPose pose;

  py::dict as_dict() const;
  M3D R() const { return pose.R(); }
  V3D T() const { return pose.T(); }

  void SetCameraId(const int input_cam_id) { cam_id = input_cam_id; }
  void SetImageName(const std::string &image_name) { image_name_ = image_name; }
  std::string image_name() const { return image_name_; }

private:
  std::string image_name_;
};

class CameraView : public CameraImage {
public:
  CameraView() {}
  CameraView(const std::string &image_name)
      : CameraImage(Camera(), image_name), cam(Camera()) {} // empty view
  CameraView(const Camera &input_cam, const std::string &image_name = "none")
      : CameraImage(input_cam, image_name), cam(input_cam) {} // empty view
  CameraView(const Camera &input_cam, const CameraPose &input_pose,
             const std::string &image_name = "none")
      : CameraImage(input_cam, input_pose, image_name), cam(input_cam) {}
  CameraView(py::dict dict);
  CameraView(const CameraView &camview)
      : CameraImage(camview), cam(camview.cam) {}

  Camera cam;
  py::array_t<uint8_t> read_image(const bool set_gray) const;

  py::dict as_dict() const;
  M3D K() const { return cam.K(); }
  M3D K_inv() const { return cam.K_inv(); }
  int h() const { return cam.h(); }
  int w() const { return cam.w(); }
  Eigen::MatrixXd matrix() const;

  V2D projection(const V3D &p3d) const;
  V3D ray_direction(const V2D &p2d) const;
  std::pair<V3D, V3D> ray_direction_gradient(const V2D &p2d) const;
  V3D get_direction_from_vp(const V3D &vp) const;

  // get focal length for initialization
  std::pair<double, bool>
  get_initial_focal_length() const; // return: (focal_length, is_prior)
};

} // namespace limap

#endif
