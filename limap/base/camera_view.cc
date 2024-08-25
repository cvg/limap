#include "base/camera_view.h"
#include <colmap/sensor/bitmap.h>

namespace limap {

CameraImage::CameraImage(py::dict dict) {
  ASSIGN_PYDICT_ITEM(dict, cam_id, int)
  pose = CameraPose(dict["pose"]);

  // load image name
  std::string image_name;
  ASSIGN_PYDICT_ITEM(dict, image_name, std::string);
  SetImageName(image_name);
}

py::dict CameraImage::as_dict() const {
  py::dict output;
  output["cam_id"] = cam_id;
  output["pose"] = pose.as_dict();
  output["image_name"] = image_name_;
  return output;
}

CameraView::CameraView(py::dict dict) {
  cam = Camera(dict["camera"]);
  pose = CameraPose(dict["pose"]);

  // load image name
  std::string image_name;
  ASSIGN_PYDICT_ITEM(dict, image_name, std::string);
  SetImageName(image_name);
}

py::dict CameraView::as_dict() const {
  py::dict output;
  output["camera"] = cam.as_dict();
  output["pose"] = pose.as_dict();
  output["image_name"] = image_name();
  return output;
}

py::array_t<uint8_t> CameraView::read_image(const bool set_gray) const {
  py::object cv2 = py::module_::import("cv2");
  py::array_t<uint8_t> img = cv2.attr("imread")(image_name());
  if (w() > 0 && h() > 0)
    img = cv2.attr("resize")(img, std::make_pair(w(), h()));
  if (set_gray) {
    img = cv2.attr("cvtColor")(img, cv2.attr("COLOR_BGR2GRAY"));
  }
  return img;
}

Eigen::MatrixXd CameraView::matrix() const {
  Eigen::MatrixXd P(3, 4);
  P.block<3, 3>(0, 0) = R();
  P.col(3) = T();
  P = K() * P;
  return P;
}

V2D CameraView::projection(const V3D &p3d) const {
  V3D p_homo = K() * (R() * p3d + T());
  V2D p2d = dehomogeneous(p_homo);
  return p2d;
}

V3D CameraView::ray_direction(const V2D &p2d) const {
  return (R().transpose() * K_inv() * homogeneous(p2d)).normalized();
}

std::pair<V3D, V3D> CameraView::ray_direction_gradient(const V2D &p2d) const {
  V3D v = R().transpose() * K_inv() * homogeneous(p2d);
  V3D direc = v.normalized();
  M3D M = (M3D::Identity() - direc * direc.transpose()) / v.norm();
  V3D grad1 = M * R().transpose() * K_inv() * V3D(1., 0., 0.);
  V3D grad2 = M * R().transpose() * K_inv() * V3D(0., 1., 0.);
  return std::make_pair(grad1, grad2);
}

V3D CameraView::get_direction_from_vp(const V3D &vp) const {
  return (R().transpose() * K_inv() * vp).normalized();
}

std::pair<double, bool> CameraView::get_initial_focal_length() const {
  colmap::Bitmap bitmap;
  double max_dim = std::max(w(), h());
  bitmap.Read(image_name());
  double ratio = 1.0;
  if (w() > 0 && h() > 0)
    ratio = max_dim / std::max(bitmap.Width(), bitmap.Height());
  double focal_length = 0.0;
  if (bitmap.ExifFocalLength(&focal_length)) {
    return std::make_pair(ratio * focal_length, true);
  } else {
    const double default_focal_length_factor = 1.2; // from COLMAP
    focal_length = default_focal_length_factor * max_dim;
    return std::make_pair(focal_length, false);
  }
}

} // namespace limap
