#pragma once

#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust cone estimation from 3D points using PoseLib RANSAC.
// Cone params: 7D [apex(3), direction(3), log_tan_alpha(1)]
// where the double cone surface is defined by:
//   angle(P - apex, direction) = alpha, tan(alpha) = exp(log_tan_alpha)
// Returns the parameter vector, or nullopt if estimation fails.
// max_error: point-to-surface distance threshold in world units.
std::optional<std::vector<double>>
EstimateConeRobust(const std::vector<V3D> &points, double max_error,
                   const poselib::RansacOptions &options,
                   poselib::RansacStats *stats = nullptr);

// Estimate cone parameters from a set of 3D points.
// Uses PCA + linear fit approach.
// Returns the parameter vector, or nullopt if estimation fails.
std::optional<std::vector<double>> EstimateCone(const std::vector<V3D> &points);

} // namespace group3d

} // namespace estimators

} // namespace limap
