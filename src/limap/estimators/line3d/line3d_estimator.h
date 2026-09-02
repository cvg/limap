#pragma once

#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>
#include <optional>

namespace limap {

namespace estimators {

namespace line3d {

// Robust 3D line estimation from points using PoseLib RANSAC.
// Returns the best-fit 3D line segment, or nullopt if estimation fails.
// max_error: point-to-line distance threshold in world units.
std::optional<Line3d>
EstimateLine3DRobust(const std::vector<V3D> &points, double max_error,
                     const poselib::RansacOptions &options,
                     poselib::RansacStats *stats = nullptr);

} // namespace line3d

} // namespace estimators

} // namespace limap
