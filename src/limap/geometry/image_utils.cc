#include "limap/geometry/image_utils.h"

namespace limap {

V3D ImageUtils::RayDirection(const colmap::Image &image, const V2D &p2d) {
  // Camera intrinsics
  const colmap::Camera &cam = RequireCamera(image);
  const Eigen::Matrix3d Kinv = cam.CalibrationMatrix().inverse();

  // World-from-camera rotation
  const Eigen::Matrix3d R_wc = WorldFromCameraRotation(image);

  // Ray in camera coords, then rotate into world, then normalize
  const V3D v_cam = Kinv * p2d.homogeneous();
  const V3D v_world = R_wc * v_cam;
  return v_world.normalized();
}

std::pair<V3D, V3D> ImageUtils::RayDirectionGradient(const colmap::Image &image,
                                                     const V2D &p2d) {
  const colmap::Camera &cam = RequireCamera(image);
  const Eigen::Matrix3d Kinv = cam.CalibrationMatrix().inverse();
  const Eigen::Matrix3d R_wc = WorldFromCameraRotation(image);

  // v = R_wc * Kinv * [x, y, 1]^T
  const V3D v = R_wc * (Kinv * p2d.homogeneous());
  const V3D d = v.normalized();

  // M = (I - d d^T) / ||v||
  const double vnorm = v.norm();
  const Eigen::Matrix3d M =
      (Eigen::Matrix3d::Identity() - d * d.transpose()) / vnorm;

  // Basis vectors for ∂/∂x and ∂/∂y in homogeneous image coords
  const V3D ex(1.0, 0.0, 0.0);
  const V3D ey(0.0, 1.0, 0.0);

  // ∂v/∂x and ∂v/∂y
  const V3D dv_dx = R_wc * (Kinv * ex);
  const V3D dv_dy = R_wc * (Kinv * ey);

  // Chain rule through normalization
  const V3D grad_x = M * dv_dx;
  const V3D grad_y = M * dv_dy;

  return {grad_x, grad_y};
}

Eigen::Matrix3x4d ImageUtils::ProjectionMatrix(const colmap::Image &image) {
  const colmap::Camera &cam = RequireCamera(image);
  const Eigen::Matrix3d K = cam.CalibrationMatrix();
  Eigen::Matrix3x4d M = image.CamFromWorld().ToMatrix();
  return K * M;
}

} // namespace limap
