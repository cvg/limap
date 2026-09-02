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
// Analytical line reprojection cost function with constant camera pose.
//
// Parameter blocks:
//   [0] line_params    (6: quaternion xyzw + SO(2) weights)
//   [1] camera_params  (CameraModel::num_params)
//
// Residual: 2D (point-to-line distance for two endpoints)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalLineReprojConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 6, CameraModel::num_params> {
public:
  AnalyticalLineReprojConstantPoseCostFunction(
      const Line2d &observed, const colmap::Rigid3d &cam_from_world)
      : observed_(observed) {
    // Precompute constant-pose quantities
    const auto &q = cam_from_world.rotation();
    const auto &t = cam_from_world.translation();
    qvec_[0] = q.x();
    qvec_[1] = q.y();
    qvec_[2] = q.z();
    qvec_[3] = q.w();
    tvec_[0] = t.x();
    tvec_[1] = t.y();
    tvec_[2] = t.z();

    R_ = q.toRotationMatrix();
    const Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(t);
    J_mcam_d_ = t_skew * R_; // d(m_cam)/dd = [t]_x * R
    J_mcam_m_ = R_;          // d(m_cam)/dm = R
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *line_params = parameters[0];
    const double *camera_params = parameters[1];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_cam = jacobians ? jacobians[1] : nullptr;
    const bool need_jac = (J_line || J_cam);

    // Stage 1: MinimalPlucker -> Plucker
    double d[3], m[3];
    double J_d_params[18], J_m_params[18]; // 3x6 each
    MinimalPluckerToPluckerWithJac(line_params, d, m,
                                   J_line ? J_d_params : nullptr,
                                   J_line ? J_m_params : nullptr);

    // Stage 2: World -> Camera (constant pose, use precomputed matrices)
    double m_cam[3];
    Eigen::Map<const Eigen::Vector3d> d_vec(d), m_vec(m);
    Eigen::Map<Eigen::Vector3d> mc(m_cam);
    mc = J_mcam_m_ * m_vec + J_mcam_d_ * d_vec;

    // Stage 3: Camera -> Image
    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    double l[3];
    double J_l_m_cam_data[9], J_l_kvec_data[12];
    LineCamToImgWithJac(kvec, m_cam, l, need_jac ? J_l_m_cam_data : nullptr,
                        J_cam ? J_l_kvec_data : nullptr);

    // Stage 4: Residual
    double J_r_l_data[6];
    LineResidualWithJac(l, observed_, residuals,
                        need_jac ? J_r_l_data : nullptr);

    // Chain Jacobians
    if (need_jac) {
      Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_r_l(
          J_r_l_data);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l_mcam(
          J_l_m_cam_data);
      const Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_r_mcam =
          J_r_l * J_l_mcam;

      if (J_line) {
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
            J_d_params);
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(
            J_m_params);

        const Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J_mcam_params =
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
  const Line2d observed_;
  double qvec_[4], tvec_[3];
  Eigen::Matrix3d R_;
  Eigen::Matrix3d J_mcam_d_; // [t]_x * R (precomputed)
  Eigen::Matrix3d J_mcam_m_; // R (precomputed)
};

////////////////////////////////////////////////////////////////////////////////
// Analytical line reprojection cost function with variable camera pose.
//
// Parameter blocks:
//   [0] line_params    (6: quaternion xyzw + SO(2) weights)
//   [1] cam_from_world (7: quaternion xyzw + translation)
//   [2] camera_params  (CameraModel::num_params)
//
// Residual: 2D (point-to-line distance for two endpoints)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalLineReprojCostFunction
    : public ceres::SizedCostFunction<2, 6, 7, CameraModel::num_params> {
public:
  explicit AnalyticalLineReprojCostFunction(const Line2d &observed)
      : observed_(observed) {}

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *line_params = parameters[0];
    const double *qvec = parameters[1];
    const double *tvec = parameters[1] + 4;
    const double *camera_params = parameters[2];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_pose = jacobians ? jacobians[1] : nullptr;
    double *J_cam = jacobians ? jacobians[2] : nullptr;
    const bool need_jac = (J_line || J_pose || J_cam);

    // Stage 1: MinimalPlucker -> Plucker
    double d[3], m[3];
    double J_d_params[18], J_m_params[18];
    MinimalPluckerToPluckerWithJac(line_params, d, m,
                                   J_line ? J_d_params : nullptr,
                                   J_line ? J_m_params : nullptr);

    // Stage 2: World -> Camera
    double m_cam[3];
    double J_mcam_qvec_data[12], J_mcam_tvec_data[9];
    double J_mcam_d_data[9], J_mcam_m_data[9];
    PluckerWorldToCamWithJac(
        qvec, tvec, d, m, m_cam, J_pose ? J_mcam_qvec_data : nullptr,
        J_pose ? J_mcam_tvec_data : nullptr, J_line ? J_mcam_d_data : nullptr,
        J_line ? J_mcam_m_data : nullptr);

    // Stage 3: Camera -> Image
    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    double l[3];
    double J_l_m_cam_data[9], J_l_kvec_data[12];
    LineCamToImgWithJac(kvec, m_cam, l, need_jac ? J_l_m_cam_data : nullptr,
                        J_cam ? J_l_kvec_data : nullptr);

    // Stage 4: Residual
    double J_r_l_data[6];
    LineResidualWithJac(l, observed_, residuals,
                        need_jac ? J_r_l_data : nullptr);

    // Chain Jacobians
    if (need_jac) {
      Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_r_l(
          J_r_l_data);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_l_mcam(
          J_l_m_cam_data);
      const Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_r_mcam =
          J_r_l * J_l_mcam;

      if (J_pose) {
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J(J_pose);
        Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J_mcam_q(
            J_mcam_qvec_data);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_mcam_t(
            J_mcam_tvec_data);
        J.leftCols<4>() = J_r_mcam * J_mcam_q;
        J.rightCols<3>() = J_r_mcam * J_mcam_t;
      }

      if (J_line) {
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
            J_d_params);
        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(
            J_m_params);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jmd(
            J_mcam_d_data);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jmm(
            J_mcam_m_data);

        const Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J_mcam_params =
            Jmd * Jd + Jmm * Jm;

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
  const Line2d observed_;
};

} // namespace estimators
} // namespace limap
