#pragma once

#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust sphere estimation from 3D points using PoseLib RANSAC.
// Sphere params: [cx, cy, cz, log_r] where r = exp(log_r).
// Using log parameterization ensures r > 0 during optimization.
// Returns (cx, cy, cz, log_r), or nullopt if estimation fails.
// max_error: point-to-surface distance threshold in world units.
std::optional<V4D> EstimateSphereRobust(const std::vector<V3D> &points,
                                        double max_error,
                                        const poselib::RansacOptions &options,
                                        poselib::RansacStats *stats = nullptr);

// Estimate sphere parameters from a set of 3D points.
// Sphere params: [cx, cy, cz, log_r] where r = exp(log_r).
// Uses algebraic distance minimization via least squares.
// Returns (cx, cy, cz, log_r), or nullopt if estimation fails.
std::optional<V4D> EstimateSphere(const std::vector<V3D> &points);

} // namespace group3d

} // namespace estimators

} // namespace limap
