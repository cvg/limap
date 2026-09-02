#pragma once

// Umbrella header for group3d estimators.
// Include this for backward compatibility or to get all estimators at once.

#include "limap/estimators/group3d/cone_estimator.h"
#include "limap/estimators/group3d/cuboid_estimator.h"
#include "limap/estimators/group3d/cylinder_estimator.h"
#include "limap/estimators/group3d/ellipsoid_estimator.h"
#include "limap/estimators/group3d/plane_estimator.h"
#include "limap/estimators/group3d/sphere_estimator.h"
#include "limap/estimators/group3d/vp_estimator.h"

#include "limap/geometry/groups.h"

#include <optional>
#include <vector>

namespace limap {
namespace estimators {
namespace group3d {

// Initialize group parameters from geometric features using RANSAC.
// This is the entry point that dispatches to the appropriate estimator.
// Returns the parameter vector if successful, nullopt otherwise.
// If init_ransac_thresh <= 0, uses GetDefaultThreshold() for the group type.
std::optional<std::vector<double>>
InitializeGroupParams(GroupType type, const std::vector<V3D> &points,
                      const std::vector<Line3d> &lines,
                      double init_ransac_thresh = -1.0);

} // namespace group3d
} // namespace estimators
} // namespace limap
