#pragma once

#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust cuboid estimation from 3D points using PoseLib RANSAC.
// Cuboid params: 10D [quat(4), d(6)]
// where the cuboid has 6 face planes defined by rotation R (from quat)
// and signed offsets d along ±x, ±y, ±z in cuboid frame.
// Returns the parameter vector, or nullopt if estimation fails.
// max_error: point-to-surface distance threshold in world units.
std::optional<std::vector<double>>
EstimateCuboidRobust(const std::vector<V3D> &points, double max_error,
                     const poselib::RansacOptions &options,
                     poselib::RansacStats *stats = nullptr);

// Estimate cuboid parameters from 3D points using PCA-based OBB fitting.
// Cuboid params: 10D [quat(4), d(6)]
// Returns the parameter vector, or nullopt if estimation fails.
std::optional<std::vector<double>>
EstimateCuboid(const std::vector<V3D> &points);

} // namespace group3d

} // namespace estimators

} // namespace limap
