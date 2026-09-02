#pragma once

#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust vanishing point estimation from 3D lines using PoseLib RANSAC.
// Returns unit 3D direction vector (||v|| = 1), or nullopt if estimation fails.
// max_error: angular error threshold in radians (angle between line direction
//            and VP direction).
std::optional<V3D> EstimateVPRobust(const std::vector<Line3d> &lines,
                                    double max_error,
                                    const poselib::RansacOptions &options,
                                    poselib::RansacStats *stats = nullptr);

// Estimate vanishing point (3D direction) from a set of 3D lines.
// Uses non-minimal least squares solver (PCA on line directions).
// Returns unit 3D direction vector (||v|| = 1), or nullopt if estimation fails.
std::optional<V3D> EstimateVP(const std::vector<Line3d> &lines);

} // namespace group3d

} // namespace estimators

} // namespace limap
