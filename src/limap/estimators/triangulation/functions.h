#pragma once

#include <colmap/scene/image.h>

#include "limap/geometry/inf_line3d.h"
#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"
#include <optional>

namespace limap {

namespace estimators {

namespace triangulation {

bool TestLineInsideRanges(const Line3d &line,
                          const std::pair<V3D, V3D> &ranges);

V3D GetNormalDirection(const Line2d &l, const colmap::Image &image);

V3D GetDirectionFromVP(const V3D &vp, const colmap::Image &image);

// weak epipolar constraints
M3D ComputeEssentialMatrix(const colmap::Rigid3d &cam1_from_world,
                           const colmap::Rigid3d &cam2_from_world);
M3D ComputeFundamentalMatrix(const colmap::Image &image1,
                             const colmap::Image &image2);

// intersect epipolar lines with the matched line on image 2
double ComputeEpipolarIoU(const Line2d &l1, const colmap::Image &image1,
                          const Line2d &l2, const colmap::Image &image2);

// point triangulation
std::optional<V3D> TriangulatePoint(const V2D &p1, const colmap::Image &image1,
                                    const V2D &p2, const colmap::Image &image2);

Eigen::Matrix3d PointTriangulationCovariance(const V2D &p1,
                                             const colmap::Image &image1,
                                             const V2D &p2,
                                             const colmap::Image &image2,
                                             const Eigen::Matrix4d &covariance);

// Triangulating endpoints for triangulation
std::optional<Line3d> TriangulateLineByEndpoints(const Line2d &l1,
                                                 const colmap::Image &image1,
                                                 const Line2d &l2,
                                                 const colmap::Image &image2);

// Asymmetric perspective to (image1, l1)
// Triangulation by plane intersection
std::optional<Line3d> TriangulateLine(const Line2d &l1,
                                      const colmap::Image &image1,
                                      const Line2d &l2,
                                      const colmap::Image &image2);

M6D LineTriangulationCovariance(const Line2d &l1, const colmap::Image &image1,
                                const Line2d &l2, const colmap::Image &image2,
                                const M8D &covariance);

// unproject endpoints with known infinite line
std::optional<Line3d>
TriangulateLineWithInfiniteLine3d(const Line2d &l1, const colmap::Image &image1,
                                  const InfiniteLine3d &inf_line);

// Asymmetric perspective to (image1, l1)
// Triangulation with a known point
std::optional<Line3d> TriangulateLineWithOnePoint(const Line2d &l1,
                                                  const colmap::Image &image1,
                                                  const Line2d &l2,
                                                  const colmap::Image &image2,
                                                  const V3D &point);

// Asymmetric perspective to (image1, l1)
// Triangulation with known direction
std::optional<Line3d> TriangulateLineWithDirection(const Line2d &l1,
                                                   const colmap::Image &image1,
                                                   const Line2d &l2,
                                                   const colmap::Image &image2,
                                                   const V3D &direction);

} // namespace triangulation

} // namespace estimators

} // namespace limap
