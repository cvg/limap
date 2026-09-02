#pragma once

#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

#include <PoseLib/types.h>

#include <optional>
#include <vector>

namespace limap {

namespace estimators {

namespace group3d {

// Robust plane estimation from 3D points and/or lines using PoseLib RANSAC.
// Line endpoints are converted to points for RANSAC.
// Plane equation: ax + by + cz + d = 0, with ||(a,b,c)|| = 1.
// d is the signed distance from origin to plane.
// Returns (a, b, c, d), or nullopt if estimation fails.
// max_error: point-to-plane distance threshold in world units.
std::optional<V4D> EstimatePlaneRobust(const std::vector<V3D> &points,
                                       const std::vector<Line3d> &lines,
                                       double max_error,
                                       const poselib::RansacOptions &options,
                                       poselib::RansacStats *stats = nullptr);

// Estimate plane parameters (a, b, c, d) from a set of 3D points.
// Plane equation: ax + by + cz + d = 0, with ||(a,b,c)|| = 1.
// d is the signed distance from origin to plane.
// Uses least squares fitting via SVD.
// Returns (a, b, c, d), or nullopt if estimation fails.
std::optional<V4D> EstimatePlaneFromPoints(const std::vector<V3D> &points);

// Estimate plane parameters from a set of 3D lines (using their endpoints).
// Returns (a, b, c, d) with ||(a,b,c)|| = 1, or nullopt if estimation fails.
std::optional<V4D> EstimatePlaneFromLines(const std::vector<Line3d> &lines);

// Estimate plane parameters from mixed 3D points and lines.
// Returns (a, b, c, d) with ||(a,b,c)|| = 1, or nullopt if estimation fails.
std::optional<V4D> EstimatePlane(const std::vector<V3D> &points,
                                 const std::vector<Line3d> &lines);

} // namespace group3d

} // namespace estimators

} // namespace limap
