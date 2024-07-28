#ifndef LIMAP_SOLVERS_TRIANGULATION_TRIANGULATE_LINE_WITH_ONE_POINT_H_
#define LIMAP_SOLVERS_TRIANGULATION_TRIANGULATE_LINE_WITH_ONE_POINT_H_

#include <Eigen/Core>

namespace limap {

namespace solvers {

namespace triangulation {

// Author: Viktor Larsson
// The input is as follow
//
// Line (nx, ny, alpha) - The 3D plane (nx, ny, nz, alpha) we want to be close
// to (projected from the other image)
//
// Point (px, py) - This is the projection of the 3D point onto the plane (the
// one we should be co-linear with)
//
// Directions (p1x, p1y) and (p2x, p2y). These are the direction vectors of the
// end-points from the reference image These should be normalized (I think, not
// sure if it matters) The backprojected points are then (lambda1*p1x,
// lambda1*p1y) and (lambda2*p2x, lambda2*p2y).
//
// We are solving for lambda. We reduce to quartic poly in mu (which is a
// lagrange multiplier, and then backsubst. to get lambda). Since there are up
// to 4 solutions for mu, we plug into the cost (distance from backproj. points
// to line, and choose the best one)
//
// Coordinate system is chosen such that (0,0) is the camera center of the
// reference view

std::pair<double, double> triangulate_line_with_one_point(
    const Eigen::Vector4d &plane, const Eigen::Vector2d &p,
    const Eigen::Vector2d &v1, const Eigen::Vector2d &v2);

} // namespace triangulation

} // namespace solvers

} // namespace limap

#endif
