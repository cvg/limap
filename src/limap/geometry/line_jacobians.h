#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <colmap/estimators/cost_functions/reprojection_error.h>
#include <colmap/geometry/rigid3.h>

#include "limap/geometry/camera_models.h"
#include "limap/geometry/line2d.h"

namespace limap {

////////////////////////////////////////////////////////////////////////////////
// Stage 1: MinimalPlucker -> Plucker with analytical Jacobians
//
// Converts 6D minimal representation [uvec(x,y,z,w), wvec(0), wvec(1)]
// to full Plucker coordinates (d[3], m[3]).
//   d = R(uvec) * e1
//   m = (wvec[1] / wvec[0]) * R(uvec) * e2
//
// No abs() needed: construction guarantees w1 > 0 and SphereManifold
// preserves this during optimization.
// Jacobian outputs are 3x6 row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void MinimalPluckerToPluckerWithJac(const double *params, double *d,
                                           double *m, double *J_d_params,
                                           double *J_m_params) {
  const double *uvec = params;
  const double w1 = params[4];
  const double w2 = params[5];
  const double b_norm = w2 / w1;

  const double e1[3] = {1.0, 0.0, 0.0};
  const double e2[3] = {0.0, 1.0, 0.0};

  // d = R(uvec) * e1, with Jacobian w.r.t. uvec
  double J_Rd_uvec[12]; // 3x4 row-major
  Eigen::Vector3d Rd = colmap::QuaternionRotatePointWithJac(
      uvec, e1, (J_d_params || J_m_params) ? J_Rd_uvec : nullptr);
  d[0] = Rd(0);
  d[1] = Rd(1);
  d[2] = Rd(2);

  // m = b_norm * R(uvec) * e2
  double J_Re2_uvec[12]; // 3x4 row-major
  Eigen::Vector3d Re2 = colmap::QuaternionRotatePointWithJac(
      uvec, e2, J_m_params ? J_Re2_uvec : nullptr);
  m[0] = b_norm * Re2(0);
  m[1] = b_norm * Re2(1);
  m[2] = b_norm * Re2(2);

  if (J_d_params) {
    // J_d_params is 3x6: [dD/duvec (3x4) | dD/dwvec (3x2)]
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(J_d_params);
    Jd.leftCols<4>() =
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(J_Rd_uvec);
    Jd.rightCols<2>().setZero();
  }

  if (J_m_params) {
    // J_m_params is 3x6: [dM/duvec (3x4) | dM/dwvec (3x2)]
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(J_m_params);
    Jm.leftCols<4>() =
        b_norm *
        Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(J_Re2_uvec);
    // dm/dw1 = -(w2 / w1^2) * Re2
    Jm.col(4) = -(w2 / (w1 * w1)) * Re2;
    // dm/dw2 = (1/w1) * Re2
    Jm.col(5) = (1.0 / w1) * Re2;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 2: Transform Plucker line from world to camera frame
//
// m_cam = R*m + [t]_x * R*d
//
// where R = R(qvec), t = tvec.
// Jacobian outputs are row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void
PluckerWorldToCamWithJac(const double *qvec, const double *tvec,
                         const double *d, const double *m, double *m_cam,
                         double *J_m_cam_qvec, // 3x4 row-major, or nullptr
                         double *J_m_cam_tvec, // 3x3 row-major, or nullptr
                         double *J_m_cam_d,    // 3x3 row-major, or nullptr
                         double *J_m_cam_m)    // 3x3 row-major, or nullptr
{
  Eigen::Map<const Eigen::Quaterniond> q(qvec);
  Eigen::Map<const Eigen::Vector3d> t(tvec);
  Eigen::Map<const Eigen::Vector3d> d_vec(d);
  Eigen::Map<const Eigen::Vector3d> m_vec(m);

  // Compute rotation matrix (used in forward pass and Jacobians)
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const Eigen::Vector3d Rd = R * d_vec;
  const Eigen::Vector3d Rm = R * m_vec;
  const Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(t);

  // Forward: m_cam = R*m + t x (R*d)
  Eigen::Map<Eigen::Vector3d> mc(m_cam);
  mc = Rm + t_skew * Rd;

  if (J_m_cam_qvec) {
    // d(m_cam)/dqvec = d(R*m)/dqvec + [t]_x * d(R*d)/dqvec
    double J_Rm_q[12], J_Rd_q[12];
    colmap::QuaternionRotatePointWithJac(qvec, m, J_Rm_q);
    colmap::QuaternionRotatePointWithJac(qvec, d, J_Rd_q);

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_m_cam_qvec);
    J = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(J_Rm_q) +
        t_skew *
            Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(J_Rd_q);
  }

  if (J_m_cam_tvec) {
    // d(t x Rd)/dt = -[Rd]_x
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_m_cam_tvec);
    J = -colmap::CrossProductMatrix(Rd);
  }

  if (J_m_cam_d) {
    // d(m_cam)/dd = [t]_x * R
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_m_cam_d);
    J = t_skew * R;
  }

  if (J_m_cam_m) {
    // d(m_cam)/dm = R
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_m_cam_m);
    J = R;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 3: Project camera-frame Plucker moment to image line
//
// l_unnorm = A * m_cam, where A = [[fy, 0, 0], [0, fx, 0],
//                                   [-cx*fy, -cy*fx, fx*fy]]
// l = l_unnorm / |l_unnorm|
//
// kvec = [fx, fy, cx, cy]
// Jacobian outputs are row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void LineCamToImgWithJac(const double *kvec, const double *m_cam,
                                double *l, double *J_l_m_cam, // 3x3
                                double *J_l_kvec)             // 3x4
{
  const double fx = kvec[0], fy = kvec[1], cx = kvec[2], cy = kvec[3];
  const double mx = m_cam[0], my = m_cam[1], mz = m_cam[2];

  // A * m_cam
  const double l0 = fy * mx;
  const double l1 = fx * my;
  const double l2 = -cx * fy * mx - cy * fx * my + fx * fy * mz;
  const double norm = std::sqrt(l0 * l0 + l1 * l1 + l2 * l2);

  if (norm < 1e-12) {
    l[0] = l[1] = l[2] = 0;
    if (J_l_m_cam) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_l_m_cam);
      J.setZero();
    }
    if (J_l_kvec) {
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_l_kvec);
      J.setZero();
    }
    return;
  }

  const Eigen::Vector3d l_unnorm(l0, l1, l2);
  const Eigen::Vector3d l_vec = l_unnorm / norm;
  l[0] = l_vec(0);
  l[1] = l_vec(1);
  l[2] = l_vec(2);

  // Normalization Jacobian: d(v/|v|)/dv = (I - l*l^T) / |v|
  const Eigen::Matrix3d J_norm =
      (Eigen::Matrix3d::Identity() - l_vec * l_vec.transpose()) / norm;

  if (J_l_m_cam) {
    // A matrix
    Eigen::Matrix3d A;
    A << fy, 0, 0, 0, fx, 0, -cx * fy, -cy * fx, fx * fy;

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_l_m_cam);
    J = J_norm * A;
  }

  if (J_l_kvec) {
    // dl_unnorm/dkvec (3x4, columns = fx, fy, cx, cy)
    Eigen::Matrix<double, 3, 4> J_unnorm_kvec;
    J_unnorm_kvec << 0, mx, 0, 0, my, 0, 0, 0, fy * mz - cy * my,
        fx * mz - cx * mx, -fy * mx, -fx * my;

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_l_kvec);
    J = J_norm * J_unnorm_kvec;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Stage 4: Compute line reprojection residual from normalized line
//
// r[i] = (a*px_i + b*py_i + c) / sqrt(a^2 + b^2)
//
// l = (a, b, c) with |l| = 1
// Jacobian output is 2x3 row-major (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void LineResidualWithJac(const double *l, const Line2d &observed,
                                double *residuals,
                                double *J_r_l) // 2x3
{
  const double a = l[0], b = l[1], c = l[2];
  const double denom_sq = a * a + b * b;
  const double denom = std::sqrt(denom_sq);

  if (denom < 1e-12) {
    residuals[0] = residuals[1] = 0;
    if (J_r_l) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_r_l);
      J.setZero();
    }
    return;
  }

  const double inv_denom = 1.0 / denom;
  const double inv_denom3 = inv_denom * inv_denom * inv_denom;
  const double px_s = observed.start.x(), py_s = observed.start.y();
  const double px_e = observed.end.x(), py_e = observed.end.y();

  const double d0 = a * px_s + b * py_s + c;
  const double d1 = a * px_e + b * py_e + c;

  residuals[0] = d0 * inv_denom;
  residuals[1] = d1 * inv_denom;

  if (J_r_l) {
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_r_l);

    // d(dist_i)/da = (px_i*(a^2+b^2) - a*d_i) / denom^3
    // d(dist_i)/db = (py_i*(a^2+b^2) - b*d_i) / denom^3
    // d(dist_i)/dc = 1 / denom
    J(0, 0) = (px_s * denom_sq - a * d0) * inv_denom3;
    J(0, 1) = (py_s * denom_sq - b * d0) * inv_denom3;
    J(0, 2) = inv_denom;

    J(1, 0) = (px_e * denom_sq - a * d1) * inv_denom3;
    J(1, 1) = (py_e * denom_sq - b * d1) * inv_denom3;
    J(1, 2) = inv_denom;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Helper: Jacobian of ParamsToKvec w.r.t. camera parameters
//
// For SimplePinhole: params=[f, cx, cy] -> kvec=[f, f, cx, cy]
// For Pinhole:       params=[fx, fy, cx, cy] -> kvec=[fx, fy, cx, cy]
////////////////////////////////////////////////////////////////////////////////
template <typename CameraModel>
inline void ParamsToKvecJac(
    Eigen::Matrix<double, 4, CameraModel::num_params> &J_kvec_params) {
  J_kvec_params.setZero();
  if constexpr (std::is_same_v<CameraModel, colmap::SimplePinholeCameraModel>) {
    J_kvec_params(0, 0) = 1.0; // fx = f
    J_kvec_params(1, 0) = 1.0; // fy = f
    J_kvec_params(2, 1) = 1.0; // cx
    J_kvec_params(3, 2) = 1.0; // cy
  } else if constexpr (std::is_same_v<CameraModel,
                                      colmap::PinholeCameraModel>) {
    J_kvec_params.setIdentity();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Point projection (camera frame → pixel) with analytical Jacobians
//
// For pinhole model: xy = [fx*X/Z + cx, fy*Y/Z + cy]
// where kvec = [fx, fy, cx, cy] and p_cam = [X, Y, Z]
//
// Jacobian outputs are row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void ImgFromCamWithJac(const double *kvec, const double *p_cam,
                              double *xy,
                              double *J_xy_pcam, // 2x3 row-major
                              double *J_xy_kvec) // 2x4 row-major
{
  const double fx = kvec[0], fy = kvec[1], cx = kvec[2], cy = kvec[3];
  const double X = p_cam[0], Y = p_cam[1], Z = p_cam[2];
  const double inv_Z = 1.0 / Z;
  const double u = X * inv_Z;
  const double v = Y * inv_Z;

  xy[0] = fx * u + cx;
  xy[1] = fy * v + cy;

  if (J_xy_pcam) {
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_xy_pcam);
    J << fx * inv_Z, 0.0, -fx * u * inv_Z, 0.0, fy * inv_Z, -fy * v * inv_Z;
  }

  if (J_xy_kvec) {
    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(J_xy_kvec);
    J << u, 0.0, 1.0, 0.0, 0.0, v, 0.0, 1.0;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Point-to-line residual with analytical Jacobians
//
// Given normalized 2D line l = [a,b,c] (|l|=1) and constant 2D point,
// compute the perpendicular distance vector residual:
//   denom = sqrt(a² + b²)
//   dist  = (a*px + b*py + c) / denom
//   r     = dist * [a, b]^T / denom
//
// Used for wireframe line-to-point costs where the projected line varies
// and the observed point is constant.
// Jacobian output is 2x3 row-major (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void PointToLineResidualWithJac(const double *l,
                                       const Eigen::Vector2d &point,
                                       double *residuals,
                                       double *J_r_l) // 2x3
{
  const double a = l[0], b = l[1], c = l[2];
  const double denom_sq = a * a + b * b;
  const double denom = std::sqrt(denom_sq);

  if (denom < 1e-12) {
    residuals[0] = residuals[1] = 0;
    if (J_r_l) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_r_l);
      J.setZero();
    }
    return;
  }

  const double inv_denom = 1.0 / denom;
  const double inv_denom3 = inv_denom * inv_denom * inv_denom;
  const double px = point.x(), py = point.y();
  const double d_raw = a * px + b * py + c;
  const double dist = d_raw * inv_denom;
  const double a_n = a * inv_denom;
  const double b_n = b * inv_denom;

  residuals[0] = dist * a_n;
  residuals[1] = dist * b_n;

  if (J_r_l) {
    // r = dist * n_hat, where n_hat = [a_n, b_n]
    // ∂r/∂l = n_hat ⊗ J_dist + dist * ∂n_hat/∂l
    const double J_dist_a = (px * denom_sq - a * d_raw) * inv_denom3;
    const double J_dist_b = (py * denom_sq - b * d_raw) * inv_denom3;
    const double J_dist_c = inv_denom;

    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(J_r_l);
    J(0, 0) = a_n * J_dist_a + dist * b * b * inv_denom3;
    J(0, 1) = a_n * J_dist_b - dist * a * b * inv_denom3;
    J(0, 2) = a_n * J_dist_c;
    J(1, 0) = b_n * J_dist_a - dist * a * b * inv_denom3;
    J(1, 1) = b_n * J_dist_b + dist * a * a * inv_denom3;
    J(1, 2) = b_n * J_dist_c;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Project a 3D point onto a Plücker line (closest point on line)
//
// Given Plücker line (d, m) with ||d|| = 1:
//   p_proj = d × m + (p · d) * d
//
// Jacobians (3×3 row-major, or nullptr to skip):
//   J_proj_d: ∂p_proj/∂d = -[m]_× + (d·p)*I + d*p^T
//                           (raw ambient Jacobian — caller composes with
//                            Stage 1 Jacobian which handles the manifold)
//   J_proj_m: ∂p_proj/∂m = [d]_×
////////////////////////////////////////////////////////////////////////////////
inline void PointOnLineWithJac(const double *d, const double *m,
                               const double *point, double *projected,
                               double *J_proj_d, // 3×3 row-major, or nullptr
                               double *J_proj_m) // 3×3 row-major, or nullptr
{
  const Eigen::Map<const Eigen::Vector3d> d_vec(d);
  const Eigen::Map<const Eigen::Vector3d> m_vec(m);
  const Eigen::Map<const Eigen::Vector3d> p_vec(point);

  // p_proj = d × m + (p · d) * d
  const double dp = d_vec.dot(p_vec);
  const Eigen::Vector3d dxm = d_vec.cross(m_vec);
  Eigen::Map<Eigen::Vector3d> proj(projected);
  proj = dxm + dp * d_vec;

  if (J_proj_d) {
    // ∂p_proj/∂d = -[m]_× + (d·p)*I + d*p^T
    //   ∂(d×m)/∂d = -[m]_×
    //   ∂((p·d)*d)/∂d = (d·p)*I + d*p^T
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_d);
    J = -colmap::CrossProductMatrix(m_vec) + dp * Eigen::Matrix3d::Identity() +
        d_vec * p_vec.transpose();
  }

  if (J_proj_m) {
    // ∂(d×m)/∂m = [d]_×
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_m);
    J = colmap::CrossProductMatrix(d_vec);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Full endpoint Jacobian w.r.t. MinimalPlucker parameters
//
// Chains Stage 1 (MinimalPlucker → Plücker) with PointOnLine projection:
//   J_endpoint_params = J_proj_d * J_d_params + J_proj_m * J_m_params  (3×6)
//
// Input:
//   params:    6D MinimalPlucker parameters [uvec(4), wvec(2)]
//   endpoint:  3D point to project onto the line (start or end of Line3d)
// Output:
//   projected: 3D closest point on line to endpoint
//   J_proj_params: 3×6 row-major Jacobian ∂projected/∂params (or nullptr)
////////////////////////////////////////////////////////////////////////////////
inline void EndpointFromMinimalPluckerWithJac(const double *params,
                                              const double *endpoint,
                                              double *projected,
                                              double *J_proj_params) // 3×6
{
  // Stage 1: MinimalPlucker → Plücker
  double d[3], m[3];
  double J_d_params[18], J_m_params[18]; // 3×6 row-major each
  MinimalPluckerToPluckerWithJac(params, d, m,
                                 J_proj_params ? J_d_params : nullptr,
                                 J_proj_params ? J_m_params : nullptr);

  // PointOnLine projection
  double J_proj_d[9], J_proj_m[9]; // 3×3 row-major each
  PointOnLineWithJac(d, m, endpoint, projected,
                     J_proj_params ? J_proj_d : nullptr,
                     J_proj_params ? J_proj_m : nullptr);

  if (J_proj_params) {
    // Chain rule: J_proj_params = J_proj_d * J_d_params + J_proj_m * J_m_params
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J(J_proj_params);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpd(J_proj_d);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpm(J_proj_m);
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(J_d_params);
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(J_m_params);
    J = Jpd * Jd + Jpm * Jm;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Plane projection of a 3D point with analytical Jacobians
//
// p_proj = p - (n·p + d) * n
// plane_params = [nx, ny, nz, d]
// Jacobian outputs are row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void PlaneProjectPointWithJac(const double *point,
                                     const double *plane_params,
                                     double *projected,
                                     double *J_proj_point, // 3x3
                                     double *J_proj_plane) // 3x4
{
  const Eigen::Map<const Eigen::Vector3d> n(plane_params);
  const double d = plane_params[3];
  const Eigen::Map<const Eigen::Vector3d> p(point);

  const double sigma = n.dot(p) + d;

  Eigen::Map<Eigen::Vector3d> proj(projected);
  proj = p - sigma * n;

  if (J_proj_point) {
    // ∂p_proj/∂p = I - n*n^T
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_point);
    J = Eigen::Matrix3d::Identity() - n * n.transpose();
  }

  if (J_proj_plane) {
    // ∂p_proj/∂n = -(σ*I + n*p^T), ∂p_proj/∂d = -n
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_proj_plane);
    J.leftCols<3>() =
        -(sigma * Eigen::Matrix3d::Identity() + n * p.transpose());
    J.col(3) = -n;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Plane projection of a Plucker line with analytical Jacobians
//
// Projects Plucker line (d, m) onto a plane [n, w]:
//   d' = d - (d·n)*n
//   m' = (1 - α²/D)*m + (αγ/D)*d + (w - δ/D)*(d×n)
//
// where α = d·n, γ = m·n, δ = (d×n)·m, D = ||d||² + ε
// Jacobian outputs are 3x3 or 3x4 row-major matrices (or nullptr to skip).
////////////////////////////////////////////////////////////////////////////////
inline void PlaneProjectLineWithJac(const double *d_in, const double *m_in,
                                    const double *plane_params, double *proj_d,
                                    double *proj_m,
                                    double *J_proj_d_d,     // 3x3, or nullptr
                                    double *J_proj_m_d,     // 3x3, or nullptr
                                    double *J_proj_m_m,     // 3x3, or nullptr
                                    double *J_proj_d_plane, // 3x4, or nullptr
                                    double *J_proj_m_plane) // 3x4, or nullptr
{
  const Eigen::Map<const Eigen::Vector3d> d(d_in);
  const Eigen::Map<const Eigen::Vector3d> m(m_in);
  const Eigen::Map<const Eigen::Vector3d> n(plane_params);
  const double w = plane_params[3];

  const double D = d.squaredNorm() + 1e-12;
  const double inv_D = 1.0 / D;
  const double alpha = d.dot(n);
  const double gamma_val = m.dot(n);
  const Eigen::Vector3d dxn = d.cross(n);
  const double delta = dxn.dot(m);

  const double coeff_m = 1.0 - alpha * alpha * inv_D;
  const double coeff_d = alpha * gamma_val * inv_D;
  const double coeff_dxn = w - delta * inv_D;

  // Forward pass
  Eigen::Map<Eigen::Vector3d> pd(proj_d);
  Eigen::Map<Eigen::Vector3d> pm(proj_m);
  pd = d - alpha * n;
  pm = coeff_m * m + coeff_d * d + coeff_dxn * dxn;

  const bool need_jac = J_proj_d_d || J_proj_m_d || J_proj_m_m ||
                        J_proj_d_plane || J_proj_m_plane;
  if (!need_jac)
    return;

  const double inv_D2 = inv_D * inv_D;

  // ∂d'/∂d = I - n*n^T
  if (J_proj_d_d) {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_d_d);
    J = Eigen::Matrix3d::Identity() - n * n.transpose();
  }

  // ∂d'/∂plane = [-(α*I + n*d^T) | 0]
  if (J_proj_d_plane) {
    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_proj_d_plane);
    J.leftCols<3>() =
        -(alpha * Eigen::Matrix3d::Identity() + n * d.transpose());
    J.col(3).setZero();
  }

  // Precompute for m' Jacobians
  const Eigen::Vector3d nxm = n.cross(m);
  const Eigen::Vector3d dxm = d.cross(m);
  const Eigen::Matrix3d n_skew = colmap::CrossProductMatrix(n);
  const Eigen::Matrix3d d_skew = colmap::CrossProductMatrix(d);

  // ∂m'/∂d (complex: involves all coefficient gradients + ∂dxn/∂d)
  if (J_proj_m_d) {
    const Eigen::Vector3d dcoeff_m_dd =
        (2.0 * alpha * inv_D2) * (alpha * d - D * n);
    const Eigen::Vector3d dcoeff_d_dd =
        (gamma_val * inv_D2) * (D * n - 2.0 * alpha * d);
    const Eigen::Vector3d dcoeff_dxn_dd = inv_D2 * (2.0 * delta * d - D * nxm);

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_m_d);
    J = m * dcoeff_m_dd.transpose() + coeff_d * Eigen::Matrix3d::Identity() +
        d * dcoeff_d_dd.transpose() + dxn * dcoeff_dxn_dd.transpose() +
        coeff_dxn * (-n_skew);
  }

  // ∂m'/∂m = coeff_m*I + (α/D)*d*n^T - (1/D)*dxn*dxn^T
  if (J_proj_m_m) {
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(J_proj_m_m);
    J = coeff_m * Eigen::Matrix3d::Identity() +
        (alpha * inv_D) * d * n.transpose() - inv_D * dxn * dxn.transpose();
  }

  // ∂m'/∂plane = [∂m'/∂n | ∂m'/∂w = dxn]
  if (J_proj_m_plane) {
    const Eigen::Vector3d dcoeff_m_dn = -(2.0 * alpha * inv_D) * d;
    const Eigen::Vector3d dcoeff_d_dn = inv_D * (gamma_val * d + alpha * m);
    const Eigen::Vector3d dcoeff_dxn_dn = inv_D * dxm;

    Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(J_proj_m_plane);
    J.leftCols<3>() = m * dcoeff_m_dn.transpose() +
                      d * dcoeff_d_dn.transpose() +
                      dxn * dcoeff_dxn_dn.transpose() + coeff_dxn * d_skew;
    J.col(3) = dxn;
  }
}

} // namespace limap
