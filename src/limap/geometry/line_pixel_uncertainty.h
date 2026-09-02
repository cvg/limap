#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/loss_function.h>

#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"

namespace limap {

// Compute median per-view pixel standard deviation for a single line track.
//
// Uses the full Jacobian chain to propagate Plücker parameter covariance
// to pixel space:
//   1. Build information matrix H = Σ_k ρ'(s_k) * J_k^T * J_k  (6×6)
//   2. Invert to get Σ_plucker = H⁻¹
//   3. Project to endpoint 3D covariance via EndpointFromMinimalPluckerWithJac
//   4. Project to per-view pixel covariance via ImgFromCamWithJac
//   5. Return median of max pixel std dev across views
//
// For each view, backprojected endpoints are computed by finding the 3D points
// on the line closest to the camera rays through the 2D observation endpoints.
// This provides view-specific uncertainty estimates at the locations each view
// actually observes, rather than using fixed global endpoints.
//
// Note: The line3d parameter is unused (kept for API compatibility).
//
// Returns:
//   >= 0: median pixel standard deviation
//   -1.0: not computable (fewer than 2 observations)
//   infinity: degenerate (information matrix rank-deficient)
double ComputeLinePixelUncertainty(
    const double *params,                             // 6D MinimalPlucker
    const Line3d &line3d,                             // unused (API compat)
    const std::vector<Eigen::Quaterniond> &rotations, // cam_from_world
    const std::vector<Eigen::Vector3d> &translations,
    const std::vector<Eigen::Vector4d> &kvecs, // [fx,fy,cx,cy]
    const std::vector<Line2d> &lines2d,
    const ceres::LossFunction *loss_function);

} // namespace limap
