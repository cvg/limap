#pragma once

#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust ellipsoid estimation from 3D points using PoseLib RANSAC.
// Ellipsoid params: 10D [quat(4), center(3), log_scales(3)]
// where the ellipsoid is (x - c)^T R^T D R (x - c) = 1
// with D = diag(1/sx^2, 1/sy^2, 1/sz^2), si = exp(log_si)
// Returns the parameter vector, or nullopt if estimation fails.
// max_error: point-to-surface distance threshold in world units.
std::optional<std::vector<double>>
EstimateEllipsoidRobust(const std::vector<V3D> &points, double max_error,
                        const poselib::RansacOptions &options,
                        poselib::RansacStats *stats = nullptr);

// Estimate ellipsoid parameters from a set of 3D points.
// Uses algebraic distance minimization via least squares.
// Returns the parameter vector, or nullopt if estimation fails.
std::optional<std::vector<double>>
EstimateEllipsoid(const std::vector<V3D> &points);

} // namespace group3d

} // namespace estimators

} // namespace limap
