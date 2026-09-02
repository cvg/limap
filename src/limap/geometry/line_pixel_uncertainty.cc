#include "limap/geometry/line_pixel_uncertainty.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <ceres/manifold.h>

#include "limap/geometry/line_jacobians.h"
#include "limap/geometry/minimal_inf_line3d.h"

namespace limap {

namespace {

// Compute the closest point on an infinite line to a camera ray.
// This is equivalent to InfiniteLine3d::Unprojection but uses raw
// rotation/translation/kvec instead of colmap::Image.
//
// Args:
//   line_point: a point on the line
//   line_dir: unit direction of the line
//   R: rotation matrix (cam_from_world)
//   t: translation (cam_from_world)
//   kvec: [fx, fy, cx, cy]
//   p2d: 2D image point
//
// Returns: 3D point on the line closest to the camera ray through p2d
Eigen::Vector3d UnprojectToLine(const Eigen::Vector3d &line_point,
                                const Eigen::Vector3d &line_dir,
                                const Eigen::Matrix3d &R,
                                const Eigen::Vector3d &t,
                                const Eigen::Vector4d &kvec,
                                const Eigen::Vector2d &p2d) {
  // Camera center in world coords: -R^T * t
  const Eigen::Vector3d cam_center = -R.transpose() * t;

  // Ray direction in camera coords (normalized)
  const double fx = kvec[0], fy = kvec[1], cx = kvec[2], cy = kvec[3];
  Eigen::Vector3d ray_cam((p2d.x() - cx) / fx, (p2d.y() - cy) / fy, 1.0);
  ray_cam.normalize();

  // Ray direction in world coords
  const Eigen::Vector3d ray_world = R.transpose() * ray_cam;

  // Find closest point on line to ray
  // Minimize |line_point + t1*line_dir - (cam_center + t2*ray_world)|^2
  // Let C0 = line_point - cam_center, C1 = line_dir, C2 = ray_world
  // System: [1, -C1·C2; -C1·C2, 1] * [t1; t2] = [-C0·C1; C0·C2]
  const Eigen::Vector3d C0 = line_point - cam_center;
  const double dot_dirs = line_dir.dot(ray_world);
  const double B1 = -C0.dot(line_dir);
  const double B2 = C0.dot(ray_world);
  const double det = 1.0 - dot_dirs * dot_dirs;

  double t_line;
  if (det < 1e-12) {
    // Lines nearly parallel - use projection onto line
    t_line = B1;
  } else {
    t_line = (B1 + dot_dirs * B2) / det;
  }

  return line_point + t_line * line_dir;
}

// Convert 3D variance to pixel variance using focal length and depth.
// var2d = var3d * f / depth
// This is a robust approximation that captures worst-case uncertainty.
double Var3dToPixelVariance(double var3d, const Eigen::Vector3d &point,
                            const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                            const Eigen::Vector4d &kvec) {
  // Transform point to camera frame
  const Eigen::Vector3d p_cam = R * point + t;
  const double depth = p_cam.z();

  if (depth <= 0) {
    return std::numeric_limits<double>::infinity();
  }

  // Use max focal length (conservative estimate)
  const double f = std::max(kvec[0], kvec[1]);

  return var3d * f / depth;
}

} // namespace

double
ComputeLinePixelUncertainty(const double *params, const Line3d & /*line3d*/,
                            const std::vector<Eigen::Quaterniond> &rotations,
                            const std::vector<Eigen::Vector3d> &translations,
                            const std::vector<Eigen::Vector4d> &kvecs,
                            const std::vector<Line2d> &lines2d,
                            const ceres::LossFunction *loss_function) {
  const size_t num_obs = rotations.size();
  if (num_obs < 2) {
    return -1.0;
  }

  // Stage 1 (once): MinimalPlucker → Plücker with Jacobians
  double d[3], m[3];
  double J_d_params[18]; // 3×6 row-major
  double J_m_params[18]; // 3×6 row-major
  MinimalPluckerToPluckerWithJac(params, d, m, J_d_params, J_m_params);

  Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jd(J_d_params);
  Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Jm(J_m_params);

  // Line geometry for backprojection
  const Eigen::Map<const Eigen::Vector3d> d_vec(d);
  const Eigen::Map<const Eigen::Vector3d> m_vec(m);
  // A point on the line: p = d × m (when d is unit)
  const Eigen::Vector3d line_point = d_vec.cross(m_vec);
  const Eigen::Vector3d line_dir = d_vec.normalized();

  // Build information matrix H (6×6) from all observations
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();

  for (size_t k = 0; k < num_obs; ++k) {
    const Eigen::Matrix3d R = rotations[k].toRotationMatrix();
    const Eigen::Vector3d &t = translations[k];
    const double *kvec = kvecs[k].data();

    // Stage 2: world→camera Plücker transform (moment only)
    // m_cam = R*m + [t]_× * R*d
    const Eigen::Matrix3d t_skew = colmap::CrossProductMatrix(t);
    const Eigen::Vector3d Rd = R * d_vec;
    const Eigen::Vector3d m_cam = R * m_vec + t_skew * Rd;

    // Jacobians of m_cam w.r.t. (d, m)
    const Eigen::Matrix3d J_mcam_d = t_skew * R; // ∂m_cam/∂d
    const Eigen::Matrix3d J_mcam_m = R;          // ∂m_cam/∂m

    // Stage 3: camera-frame moment → normalized image line
    double l[3];
    double J_l_mcam[9]; // 3×3 row-major
    LineCamToImgWithJac(kvec, m_cam.data(), l, J_l_mcam, nullptr);

    // Stage 4: line residual
    double residuals[2];
    double J_r_l[6]; // 2×3 row-major
    LineResidualWithJac(l, lines2d[k], residuals, J_r_l);

    // Chain: J_k = J_r_l * J_l_mcam * (J_mcam_d * J_d + J_mcam_m * J_m) (2×6)
    Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jr(J_r_l);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jl(J_l_mcam);
    Eigen::Matrix<double, 2, 3> J_r_mcam = Jr * Jl;
    Eigen::Matrix<double, 2, 6> J_k =
        J_r_mcam * (J_mcam_d * Jd + J_mcam_m * Jm);

    // Loss weighting: ρ'(s_k) where s_k = r[0]² + r[1]²
    double weight = 1.0;
    if (loss_function) {
      double s_k = residuals[0] * residuals[0] + residuals[1] * residuals[1];
      double rho[3];
      loss_function->Evaluate(s_k, rho);
      weight = rho[1]; // ρ'(s_k)
      if (weight <= 0) {
        continue; // negative weight means observation is fully rejected
      }
    }

    // Accumulate: H += ρ'(s_k) * J_k^T * J_k
    H.noalias() += weight * J_k.transpose() * J_k;
  }

  // Project information matrix onto manifold tangent space.
  // MinimalPlucker lives on SO(3) × S¹ (4 DOF in 6D ambient space).
  // The ambient 6×6 H is rank ≤ 4, so we must invert in tangent space.
  MinimalInfiniteLine3dManifold manifold;
  Eigen::Matrix<double, 6, 4, Eigen::RowMajor> B;
  manifold.MinusJacobian(params, B.data());

  Eigen::Matrix4d H_tangent = B.transpose() * H * B;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(H_tangent);
  if (solver.info() != Eigen::Success) {
    return std::numeric_limits<double>::infinity();
  }
  const auto &eigenvalues = solver.eigenvalues();
  const double min_eigenvalue = eigenvalues.minCoeff();
  if (min_eigenvalue < 1e-10) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Matrix4d cov_tangent = solver.eigenvectors() *
                                      eigenvalues.cwiseInverse().asDiagonal() *
                                      solver.eigenvectors().transpose();

  // Lift back to ambient covariance (rank 4, but correct for projection)
  const Eigen::Matrix<double, 6, 6> cov_plucker =
      B * cov_tangent * B.transpose();

  // Compute 3D endpoint covariances and extract max singular values
  std::vector<double> sigma_per_view;
  sigma_per_view.reserve(num_obs);

  for (size_t k = 0; k < num_obs; ++k) {
    const Eigen::Matrix3d R = rotations[k].toRotationMatrix();
    const Eigen::Vector3d &t = translations[k];

    // Compute backprojected endpoints for THIS view
    const Eigen::Vector3d backproj_start =
        UnprojectToLine(line_point, line_dir, R, t, kvecs[k], lines2d[k].start);
    const Eigen::Vector3d backproj_end =
        UnprojectToLine(line_point, line_dir, R, t, kvecs[k], lines2d[k].end);

    // Compute 3D endpoint covariances via Jacobian propagation
    double proj_start[3], proj_end[3];
    double J_start[18], J_end[18]; // 3×6 row-major each
    EndpointFromMinimalPluckerWithJac(params, backproj_start.data(), proj_start,
                                      J_start);
    EndpointFromMinimalPluckerWithJac(params, backproj_end.data(), proj_end,
                                      J_end);

    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Js(J_start);
    Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> Je(J_end);
    const Eigen::Matrix3d cov_start_3d = Js * cov_plucker * Js.transpose();
    const Eigen::Matrix3d cov_end_3d = Je * cov_plucker * Je.transpose();

    // Extract max singular value (worst-case 3D std dev) for each endpoint
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svd_start(cov_start_3d);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> svd_end(cov_end_3d);

    if (svd_start.info() != Eigen::Success ||
        svd_end.info() != Eigen::Success) {
      continue;
    }

    const double var3d_start =
        std::sqrt(std::max(0.0, svd_start.eigenvalues().maxCoeff()));
    const double var3d_end =
        std::sqrt(std::max(0.0, svd_end.eigenvalues().maxCoeff()));

    // Convert to pixel variance using f/depth formula
    const double var2d_start = Var3dToPixelVariance(
        var3d_start, Eigen::Map<const Eigen::Vector3d>(proj_start), R, t,
        kvecs[k]);
    const double var2d_end = Var3dToPixelVariance(
        var3d_end, Eigen::Map<const Eigen::Vector3d>(proj_end), R, t, kvecs[k]);

    if (std::isfinite(var2d_start) && std::isfinite(var2d_end)) {
      sigma_per_view.push_back(std::max(var2d_start, var2d_end));
    }
  }

  if (sigma_per_view.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  // Return median
  const size_t n = sigma_per_view.size();
  auto mid = sigma_per_view.begin() + n / 2;
  std::nth_element(sigma_per_view.begin(), mid, sigma_per_view.end());
  if (n % 2 == 1) {
    return *mid;
  } else {
    double upper = *mid;
    auto lower_it = std::max_element(sigma_per_view.begin(), mid);
    return 0.5 * (*lower_it + upper);
  }
}

} // namespace limap
