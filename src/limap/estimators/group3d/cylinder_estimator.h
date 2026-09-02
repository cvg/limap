#pragma once

#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust cylinder estimation from 3D points using PoseLib RANSAC.
// Cylinder params: 7D [quat(4), wvec(2), log_r(1)] using MinimalInfiniteLine3d
// representation for the axis, plus log-radius.
// Returns the parameter vector, or nullopt if estimation fails.
// max_error: point-to-surface distance threshold in world units.
std::optional<std::vector<double>>
EstimateCylinderRobust(const std::vector<V3D> &points, double max_error,
                       const poselib::RansacOptions &options,
                       poselib::RansacStats *stats = nullptr);

// Estimate cylinder parameters from a set of 3D points.
// Uses iterative refinement starting from an initial guess.
// Returns the parameter vector, or nullopt if estimation fails.
std::optional<std::vector<double>>
EstimateCylinder(const std::vector<V3D> &points);

} // namespace group3d

} // namespace estimators

} // namespace limap
