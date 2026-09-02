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
// Analytical VP cost: sine(angle) between line direction and VP direction
//
// Residual: 1D  r = ||d × vp||  (both d and vp are unit vectors)
// Parameter blocks:
//   [0] line_params  (6: quaternion xyzw + SO(2) weights)
//   [1] vp_params    (3: unit vector, with SphereManifold<3>)
////////////////////////////////////////////////////////////////////////////////
class AnalyticalLineToVPSineCostFunction
    : public ceres::SizedCostFunction<1, 6, 3> {
public:
  AnalyticalLineToVPSineCostFunction() = default;

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *line_params = parameters[0];
    const double *vp_params = parameters[1];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_vp = jacobians ? jacobians[1] : nullptr;
    const bool need_jac = (J_line || J_vp);

    // Stage 1: MinimalPlucker -> direction vector d
    double d[3], m[3];
    double J_d_params[18]; // 3x6 row-major
    MinimalPluckerToPluckerWithJac(line_params, d, m,
                                   J_line ? J_d_params : nullptr, nullptr);

    // Compute cross product c = d × vp
    const Eigen::Map<const Eigen::Vector3d> d_vec(d);
    const Eigen::Map<const Eigen::Vector3d> vp_vec(vp_params);
    const Eigen::Vector3d c = d_vec.cross(vp_vec);
    const double n = c.norm();

    residuals[0] = n;

    if (!need_jac)
      return true;

    if (n < 1e-12) {
      // Degenerate: d parallel to vp, Jacobian undefined
      if (J_line) {
        Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(J_line);
        J.setZero();
      }
      if (J_vp) {
        Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J(J_vp);
        J.setZero();
      }
      return true;
    }

    // ∂r/∂d = c^T/n · (-[vp]_×) and ∂r/∂vp = c^T/n · [d]_×
    const Eigen::RowVector3d c_over_n = c.transpose() / n;

    if (J_line) {
      // ∂r/∂d (1×3) then chain with J_d_params (3×6)
      const Eigen::Matrix3d neg_vp_skew = -colmap::CrossProductMatrix(vp_vec);
      const Eigen::RowVector3d dr_dd = c_over_n * neg_vp_skew;

      Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
          J_d_params);
      Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(J_line);
      J = dr_dd * Jd;
    }

    if (J_vp) {
      // ∂r/∂vp (1×3) — Ceres handles SphereManifold projection
      const Eigen::Matrix3d d_skew = colmap::CrossProductMatrix(d_vec);
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J(J_vp);
      J = c_over_n * d_skew;
    }

    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Analytical VP cost: 1 - |cos(angle)| between line direction and VP direction
//
// Residual: 1D  r = 1 - |d · vp|  (both d and vp are unit vectors)
// Parameter blocks:
//   [0] line_params  (6: quaternion xyzw + SO(2) weights)
//   [1] vp_params    (3: unit vector, with SphereManifold<3>)
////////////////////////////////////////////////////////////////////////////////
class AnalyticalLineToVPCosineCostFunction
    : public ceres::SizedCostFunction<1, 6, 3> {
public:
  AnalyticalLineToVPCosineCostFunction() = default;

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *line_params = parameters[0];
    const double *vp_params = parameters[1];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_vp = jacobians ? jacobians[1] : nullptr;
    const bool need_jac = (J_line || J_vp);

    // Stage 1: MinimalPlucker -> direction vector d
    double d[3], m[3];
    double J_d_params[18]; // 3x6 row-major
    MinimalPluckerToPluckerWithJac(line_params, d, m,
                                   J_line ? J_d_params : nullptr, nullptr);

    // Compute dot product and absolute cosine
    const Eigen::Map<const Eigen::Vector3d> d_vec(d);
    const Eigen::Map<const Eigen::Vector3d> vp_vec(vp_params);
    const double dot = d_vec.dot(vp_vec);
    const double abs_dot = std::abs(dot);
    const double s = (dot >= 0) ? 1.0 : -1.0;

    residuals[0] = 1.0 - abs_dot;

    if (!need_jac)
      return true;

    // ∂r/∂d = -s * vp^T,  ∂r/∂vp = -s * d^T
    if (J_line) {
      const Eigen::RowVector3d dr_dd = -s * vp_vec.transpose();

      Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
          J_d_params);
      Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J(J_line);
      J = dr_dd * Jd;
    }

    if (J_vp) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> J(J_vp);
      J = -s * d_vec.transpose();
    }

    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Analytical Plane: PointToPlane2D with constant pose
//
// Residual: 2D  r = pixel(point) - pixel(PlaneProject(point, plane))
// Parameter blocks:
//   [0] point3D        (3)
//   [1] plane_params   (4: [nx, ny, nz, d])
//   [2] camera_params  (CameraModel::num_params)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalPointToPlane2DConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 3, 4, CameraModel::num_params> {
public:
  explicit AnalyticalPointToPlane2DConstantPoseCostFunction(
      const colmap::Rigid3d &cam_from_world) {
    const auto &q = cam_from_world.rotation();
    const auto &t = cam_from_world.translation();
    R_ = q.toRotationMatrix();
    t_ = t;
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const double *point3D = parameters[0];
    const double *plane_params = parameters[1];
    const double *camera_params = parameters[2];

    double *J_point = jacobians ? jacobians[0] : nullptr;
    double *J_plane = jacobians ? jacobians[1] : nullptr;
    double *J_cam = jacobians ? jacobians[2] : nullptr;
    const bool need_jac = (J_cam || J_point || J_plane);

    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    // Transform original point to camera frame
    const Eigen::Map<const Eigen::Vector3d> p(point3D);
    const Eigen::Vector3d p_cam = R_ * p + t_;

    // Project original point to pixel
    double xy_orig[2];
    double J_xy_orig_pcam[6], J_xy_orig_kvec[8]; // 2x3, 2x4
    ImgFromCamWithJac(kvec, p_cam.data(), xy_orig,
                      need_jac ? J_xy_orig_pcam : nullptr,
                      J_cam ? J_xy_orig_kvec : nullptr);

    // Project point onto plane (in world frame)
    double projected[3];
    double J_proj_point_data[9], J_proj_plane_data[12]; // 3x3, 3x4
    PlaneProjectPointWithJac(point3D, plane_params, projected,
                             (J_point || J_cam) ? J_proj_point_data : nullptr,
                             J_plane ? J_proj_plane_data : nullptr);

    // Transform projected point to camera frame
    const Eigen::Map<const Eigen::Vector3d> p_proj(projected);
    const Eigen::Vector3d p_proj_cam = R_ * p_proj + t_;

    // Project projected point to pixel
    double xy_proj[2];
    double J_xy_proj_pcam[6], J_xy_proj_kvec[8]; // 2x3, 2x4
    ImgFromCamWithJac(kvec, p_proj_cam.data(), xy_proj,
                      need_jac ? J_xy_proj_pcam : nullptr,
                      J_cam ? J_xy_proj_kvec : nullptr);

    // Residual: r = xy_orig - xy_proj
    residuals[0] = xy_orig[0] - xy_proj[0];
    residuals[1] = xy_orig[1] - xy_proj[1];

    if (!need_jac)
      return true;

    // Map Jacobians
    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_xy_orig_pc(
        J_xy_orig_pcam);
    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_xy_proj_pc(
        J_xy_proj_pcam);

    if (J_point) {
      // ∂r/∂p = J_orig_pcam * R - J_proj_pcam * R * ∂p_proj/∂p
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_proj_p(
          J_proj_point_data);
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_point);
      J = J_xy_orig_pc * R_ - J_xy_proj_pc * R_ * J_proj_p;
    }

    if (J_plane) {
      // ∂r/∂plane = -J_proj_pcam * R * ∂p_proj/∂plane
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J_proj_pl(
          J_proj_plane_data);
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(J_plane);
      J = -J_xy_proj_pc * R_ * J_proj_pl;
    }

    if (J_cam) {
      // ∂r/∂camera = (J_xy_orig_kvec - J_xy_proj_kvec) * J_kvec_params
      Eigen::Map<const Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_orig_kv(
          J_xy_orig_kvec);
      Eigen::Map<const Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_proj_kv(
          J_xy_proj_kvec);
      Eigen::Matrix<double, 4, CameraModel::num_params> J_kvec_params;
      ParamsToKvecJac<CameraModel>(J_kvec_params);
      Eigen::Map<
          Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
          J(J_cam);
      J = (J_orig_kv - J_proj_kv) * J_kvec_params;
    }

    return true;
  }

private:
  Eigen::Matrix3d R_;
  Eigen::Vector3d t_;
};

////////////////////////////////////////////////////////////////////////////////
// Analytical Plane: LineToPlane2D with constant pose
//
// Residual: 2D  r = LineReproj(line) - LineReproj(PlaneProjectLine(line,
// plane)) Parameter blocks:
//   [0] line_params    (6: quaternion xyzw + SO(2) weights)
//   [1] plane_params   (4: [nx, ny, nz, d])
//   [2] camera_params  (CameraModel::num_params)
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
class AnalyticalLineToPlane2DConstantPoseCostFunction
    : public ceres::SizedCostFunction<2, 6, 4, CameraModel::num_params> {
public:
  AnalyticalLineToPlane2DConstantPoseCostFunction(
      const Line2d &observed, const colmap::Rigid3d &cam_from_world)
      : observed_(observed) {
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
    const double *plane_params = parameters[1];
    const double *camera_params = parameters[2];

    double *J_line = jacobians ? jacobians[0] : nullptr;
    double *J_plane = jacobians ? jacobians[1] : nullptr;
    double *J_cam = jacobians ? jacobians[2] : nullptr;
    const bool need_jac = (J_cam || J_line || J_plane);

    // Stage 1: MinimalPlucker -> Plucker
    double d[3], m_data[3];
    double J_d_params[18], J_m_params[18]; // 3x6 each
    MinimalPluckerToPluckerWithJac(line_params, d, m_data,
                                   J_line ? J_d_params : nullptr,
                                   J_line ? J_m_params : nullptr);

    double kvec[4];
    ParamsToKvec<double>(CameraModel::model_id, camera_params, kvec);

    //==========================================================================
    // Original line reprojection (stages 2-4)
    //==========================================================================
    double m_cam_orig[3];
    Eigen::Map<const Eigen::Vector3d> d_vec(d), m_vec(m_data);
    Eigen::Map<Eigen::Vector3d> mc_orig(m_cam_orig);
    mc_orig = J_mcam_m_ * m_vec + J_mcam_d_ * d_vec;

    double l_orig[3];
    double J_l_orig_mcam[9], J_l_orig_kvec[12]; // 3x3, 3x4
    LineCamToImgWithJac(kvec, m_cam_orig, l_orig,
                        need_jac ? J_l_orig_mcam : nullptr,
                        J_cam ? J_l_orig_kvec : nullptr);

    double r_orig[2];
    double J_r_orig_l[6]; // 2x3
    LineResidualWithJac(l_orig, observed_, r_orig,
                        need_jac ? J_r_orig_l : nullptr);

    //==========================================================================
    // Plane-projected line reprojection (stages 5-8)
    //==========================================================================
    double proj_d[3], proj_m[3];
    double J_pd_d[9], J_pm_d[9], J_pm_m[9]; // 3x3 each
    double J_pd_plane[12], J_pm_plane[12];  // 3x4 each
    PlaneProjectLineWithJac(
        d, m_data, plane_params, proj_d, proj_m,
        (J_line || J_plane) ? J_pd_d : nullptr, J_line ? J_pm_d : nullptr,
        J_line ? J_pm_m : nullptr, J_plane ? J_pd_plane : nullptr,
        J_plane ? J_pm_plane : nullptr);

    double m_cam_surf[3];
    Eigen::Map<const Eigen::Vector3d> pd_vec(proj_d), pm_vec(proj_m);
    Eigen::Map<Eigen::Vector3d> mc_surf(m_cam_surf);
    mc_surf = J_mcam_m_ * pm_vec + J_mcam_d_ * pd_vec;

    double l_surf[3];
    double J_l_surf_mcam[9], J_l_surf_kvec[12]; // 3x3, 3x4
    LineCamToImgWithJac(kvec, m_cam_surf, l_surf,
                        need_jac ? J_l_surf_mcam : nullptr,
                        J_cam ? J_l_surf_kvec : nullptr);

    double r_surf[2];
    double J_r_surf_l[6]; // 2x3
    LineResidualWithJac(l_surf, observed_, r_surf,
                        need_jac ? J_r_surf_l : nullptr);

    // Final residual
    residuals[0] = r_orig[0] - r_surf[0];
    residuals[1] = r_orig[1] - r_surf[1];

    if (!need_jac)
      return true;

    //==========================================================================
    // Chain Jacobians
    //==========================================================================

    // Original path: J_r_orig -> J_l_orig -> J_mcam_orig
    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr_orig_l(
        J_r_orig_l);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jl_orig_mc(
        J_l_orig_mcam);
    const Eigen::Matrix<double, 2, 3> Jr_orig_mc = Jr_orig_l * Jl_orig_mc;

    // Surface path: J_r_surf -> J_l_surf -> J_mcam_surf
    Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr_surf_l(
        J_r_surf_l);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jl_surf_mc(
        J_l_surf_mcam);
    const Eigen::Matrix<double, 2, 3> Jr_surf_mc = Jr_surf_l * Jl_surf_mc;

    if (J_line) {
      // Original: ∂r_orig/∂line_params
      Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(
          J_d_params);
      Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(
          J_m_params);
      const Eigen::Matrix<double, 3, 6> J_mcam_orig_params =
          J_mcam_d_ * Jd + J_mcam_m_ * Jm;
      const Eigen::Matrix<double, 2, 6> Jr_orig_line =
          Jr_orig_mc * J_mcam_orig_params;

      // Surface: ∂r_surf/∂line_params via plane projection
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpd_d(
          J_pd_d);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpm_d(
          J_pm_d);
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpm_m(
          J_pm_m);

      // ∂m_cam_surf/∂d = J_mcam_d*∂d'/∂d + J_mcam_m*∂m'/∂d
      const Eigen::Matrix3d J_mc_surf_d = J_mcam_d_ * Jpd_d + J_mcam_m_ * Jpm_d;
      // ∂m_cam_surf/∂m = J_mcam_m*∂m'/∂m (∂d'/∂m = 0)
      const Eigen::Matrix3d J_mc_surf_m = J_mcam_m_ * Jpm_m;

      const Eigen::Matrix<double, 3, 6> J_mcam_surf_params =
          J_mc_surf_d * Jd + J_mc_surf_m * Jm;
      const Eigen::Matrix<double, 2, 6> Jr_surf_line =
          Jr_surf_mc * J_mcam_surf_params;

      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(J_line);
      J = Jr_orig_line - Jr_surf_line;
    }

    if (J_plane) {
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jpd_pl(
          J_pd_plane);
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jpm_pl(
          J_pm_plane);

      // ∂m_cam_surf/∂plane = J_mcam_d*∂d'/∂plane + J_mcam_m*∂m'/∂plane
      const Eigen::Matrix<double, 3, 4> J_mc_surf_plane =
          J_mcam_d_ * Jpd_pl + J_mcam_m_ * Jpm_pl;

      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(J_plane);
      J = -(Jr_surf_mc * J_mc_surf_plane);
    }

    if (J_cam) {
      // ∂r/∂camera = (J_r_orig_l * J_l_orig_kvec - J_r_surf_l * J_l_surf_kvec)
      //              * J_kvec_params
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jl_orig_kv(
          J_l_orig_kvec);
      Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> Jl_surf_kv(
          J_l_surf_kvec);

      Eigen::Matrix<double, 4, CameraModel::num_params> J_kvec_params;
      ParamsToKvecJac<CameraModel>(J_kvec_params);

      Eigen::Map<
          Eigen::Matrix<double, 2, CameraModel::num_params, Eigen::RowMajor>>
          J(J_cam);
      J = (Jr_orig_l * Jl_orig_kv - Jr_surf_l * Jl_surf_kv) * J_kvec_params;
    }

    return true;
  }

private:
  const Line2d observed_;
  Eigen::Matrix3d R_;
  Eigen::Matrix3d J_mcam_d_; // [t]_x * R
  Eigen::Matrix3d J_mcam_m_; // R
};

} // namespace estimators
} // namespace limap
