#pragma once

#include <ceres/ceres.h>
#include <colmap/geometry/rigid3.h>
#include <colmap/sensor/models.h>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/line2d.h"
#include "limap/geometry/line_jacobians.h"

namespace limap {
namespace estimators {

////////////////////////////////////////////////////////////////////////////////
// Analytical wireframe: LineToPoint with constant pose
//
// Project 3D line to 2D line, measure perpendicular distance from observed
// 2D point to the projected line.
//
// Residual: 2D  r = PointToLineDist(observed_point, projected_line)
// Parameter blocks:
//   [0] line_params    (6: quaternion xyzw + SO(2) weights)
//   [1] camera_params  (CameraModel::num_params)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalLineToPointConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 6, CameraModel::num_params> {
public:
  AnalyticalLineToPointConstantPoseCostFunction(
      const Eigen::Vector2d &observed_point,
      const colmap::Rigid3d &cam_from_world)
      : observed_point_(observed_point) {
    const auto &q = cam_from_world.rotation();
    const auto &t = cam_from_world.translation();
    R_ = q.toRotationMatrix();
    const Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(t);
    J_mcam_d_ = t_skew * R_;
    J_mcam_m_ = R_;
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *line_params = parameters[0];
    const double *camera_params = parameters[1];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_cam = jacobians ? jacobians[1] : nullptr;
    const bool need_jac = (J_cam || J_line);

    // Stage 1: MinimalPlucker -> Plucker
    double d[3], m[3];
    double J_d_params[18], J_m_params[18]; // 3x6 each
    MinimalPluckerToPluckerWithJac(line_params, d, m,
                                   J_line ? J_d_params : nullptr,
                                   J_line ? J_m_params : nullptr);

    // Stage 2: World -> Camera (constant pose, precomputed matrices)
    double m_cam[3];
    Eigen::Map<const Eigen::Vector3d> d_vec(d), m_vec(m);
    Eigen::Map<Eigen::Vector3d> mc(m_cam);
    mc = J_mcam_m_ * m_vec + J_mcam_d_ * d_vec;

    // Stage 3: Camera -> Image
    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    double l[3];
    double J_l_m_cam_data[9], J_l_kvec_data[12]; // 3x3, 3x4
    LineCamToImgWithJac(kvec, m_cam, l, need_jac ? J_l_m_cam_data : nullptr,
                        J_cam ? J_l_kvec_data : nullptr);

    // Stage 4: Point-to-line residual (observed point to projected line)
    double J_r_l_data[6]; // 2x3
    PointToLineResidualWithJac(l, observed_point_, residuals,
                               need_jac ? J_r_l_data : nullptr);

    // Chain Jacobians
    if (need_jac) {
      Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_r_l(
          J_r_l_data);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l_mcam(
          J_l_m_cam_data);
      const Eigen::Matrix<double, 2, 3> J_r_mcam = J_r_l * J_l_mcam;

      if (J_line) {
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
            J_d_params);
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(
            J_m_params);

        const Eigen::Matrix<double, 3, 6> J_mcam_params =
            J_mcam_d_ * Jd + J_mcam_m_ * Jm;

        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(J_line);
        J = J_r_mcam * J_mcam_params;
      }

      if (J_cam) {
        Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J_l_kvec(
            J_l_kvec_data);

        Eigen::Matrix<double, 4, CameraModel::num_params> J_kvec_params;
        ParamsToKvecJac<CameraModel>(J_kvec_params);

        Eigen::Map<
            Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
            J(J_cam);
        J = J_r_l * J_l_kvec * J_kvec_params;
      }
    }

    return true;
  }

private:
  const Eigen::Vector2d observed_point_;
  Eigen::Matrix3d R_;
  Eigen::Matrix3d J_mcam_d_;
  Eigen::Matrix3d J_mcam_m_;
};

////////////////////////////////////////////////////////////////////////////////
// Analytical wireframe: PointToLine with constant pose
//
// Project 3D point to 2D pixel, measure perpendicular distance to observed
// 2D line.
//
// Residual: 2D  r = PointToLineDist(projected_pixel, observed_line)
// Parameter blocks:
//   [0] point3D        (3)
//   [1] camera_params  (CameraModel::num_params)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalPointToLineConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 3, CameraModel::num_params> {
public:
  AnalyticalPointToLineConstantPoseCostFunction(
      const Line2d &observed_line, const colmap::Rigid3d &cam_from_world)
      : cam_from_world_(cam_from_world) {
    R_ = cam_from_world.rotation().toRotationMatrix();
    t_ = cam_from_world.translation();

    // Pre-normalize the observed line to [a, b, c] with sqrt(a²+b²) = 1
    const double a_raw = observed_line.end.y() - observed_line.start.y();
    const double b_raw = observed_line.start.x() - observed_line.end.x();
    const double c_raw = observed_line.end.x() * observed_line.start.y() -
                         observed_line.start.x() * observed_line.end.y();
    const double denom = std::sqrt(a_raw * a_raw + b_raw * b_raw);
    if (denom > 1e-12) {
      a_ = a_raw / denom;
      b_ = b_raw / denom;
      c_ = c_raw / denom;
    } else {
      a_ = b_ = c_ = 0;
    }

    // Precompute J_r_xy (2x2 constant matrix) = n_hat * n_hat^T
    J_r_xy_ << a_ * a_, a_ * b_, a_ * b_, b_ * b_;
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *point3D = parameters[0];
    const double *camera_params = parameters[1];

    double *J_point = jacobians ? jacobians[0] : nullptr;
    double *J_cam = jacobians ? jacobians[1] : nullptr;
    const bool need_jac = (J_cam || J_point);

    // Transform to camera frame (constant pose)
    const Eigen::Map<const Eigen::Vector3d> p(point3D);
    const Eigen::Vector3d p_cam = R_ * p + t_;

    // Project to pixel
    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    double xy[2];
    double J_xy_pcam_data[6], J_xy_kvec_data[8]; // 2x3, 2x4
    ImgFromCamWithJac(kvec, p_cam.data(), xy,
                      J_point ? J_xy_pcam_data : nullptr,
                      J_cam ? J_xy_kvec_data : nullptr);

    // Compute distance from pixel to observed line
    const double d_signed = a_ * xy[0] + b_ * xy[1] + c_;
    residuals[0] = d_signed * a_;
    residuals[1] = d_signed * b_;

    if (!need_jac)
      return true;

    if (J_point) {
      // ∂r/∂point = J_r_xy * J_xy_pcam * R
      Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_xy_pcam(
          J_xy_pcam_data);
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_point);
      J = J_r_xy_ * J_xy_pcam * R_;
    }

    if (J_cam) {
      // ∂r/∂camera = J_r_xy * J_xy_kvec * J_kvec_params
      Eigen::Map<const Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_xy_kvec(
          J_xy_kvec_data);
      Eigen::Matrix<double, 4, CameraModel::num_params> J_kvec_params;
      ParamsToKvecJac<CameraModel>(J_kvec_params);
      Eigen::Map<
          Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
          J(J_cam);
      J = J_r_xy_ * J_xy_kvec * J_kvec_params;
    }

    return true;
  }

private:
  const colmap::Rigid3d cam_from_world_;
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
  double a_, b_, c_;
  Eigen::Matrix2d J_r_xy_; // constant: n_hat * n_hat^T
};

////////////////////////////////////////////////////////////////////////////////
// Factory functions for analytical wireframe costs (camera model dispatch)
////////////////////////////////////////////////////////////////////////////////

inline ceres::CostFunction *
CreateAnalyticalLineToPointConstantPose(colmap::CameraModelId camera_model_id,
                                        const Eigen::Vector2d &observed_point,
                                        const colmap::Rigid3d &cam_from_world) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return new AnalyticalLineToPointConstantPoseCostFunction<CameraModel>(     \
        observed_point, cam_from_world);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error(
        "Camera model not supported for analytical wireframe projection");
  }
}

inline ceres::CostFunction *
CreateAnalyticalPointToLineConstantPose(colmap::CameraModelId camera_model_id,
                                        const Line2d &observed_line,
                                        const colmap::Rigid3d &cam_from_world) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::model_id:                                                  \
    return new AnalyticalPointToLineConstantPoseCostFunction<CameraModel>(     \
        observed_line, cam_from_world);

    LIMAP_UNDISTORTED_CAMERA_MODEL_CASES

#undef CAMERA_MODEL_CASE
  default:
    throw std::domain_error(
        "Camera model not supported for analytical wireframe projection");
  }
}

} // namespace estimators
} // namespace limap
