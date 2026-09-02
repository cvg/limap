#pragma once

#include <colmap/scene/image.h>

#include "limap/geometry/inf_line2d.h"
#include "limap/geometry/line2d.h"
#include "limap/geometry/line3d.h"
#include "limap/util/eigen_types.h"

namespace limap {

// InfiniteLine3d with Plucker coordinate
class InfiniteLine3d {
public:
  InfiniteLine3d() = default;

  // use_normal == true  -> (a, b) = (point, direction) with unit direction
  // use_normal == false -> (a, b) = (direction, moment) Plücker pair
  InfiniteLine3d(const V3D &a, const V3D &b, bool use_normal = true);
  explicit InfiniteLine3d(const Line3d &line);

  V3D PointProjection(const V3D &q) const;
  double PointDistance(const V3D &q) const;

  InfiniteLine2d Projection(const colmap::Image &image) const;

  // find the closest 3D point on this line to the camera ray
  // that passes through the 2D image point p2d
  V3D Unprojection(const V2D &p2d, const colmap::Image &image) const;

  V3D ProjectFromInfiniteLine(const InfiniteLine3d &line) const;
  V3D ProjectToInfiniteLine(const InfiniteLine3d &line) const;

  V3D Point() const;     // a concrete point on the line
  V3D Direction() const; // direction (as stored)
  M4D Matrix() const;    // Plücker matrix

  // Data
  V3D d; // direction
  V3D m; // moment = p × d
};

// Estimate a finite 3D line segment along an infinite 3D line
// using multiple 2D line observations from different images.
// 'images.size()' must match 'line2ds.size()'.
// Outliers are trimmed using the num_outliers parameter.
Line3d GetLineSegmentFromInfiniteLine3d(
    const InfiniteLine3d &inf_line, const std::vector<colmap::Image> &images,
    const std::vector<Line2d> &line2ds,
    const int num_outliers = 2); // views.size() == line2ds.size()

// Estimate a finite 3D line segment along an infinite 3D line
// using multiple 3D line observations. Outliers are trimmed by num_outliers.
Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d &inf_line,
                                        const std::vector<Line3d> &line3ds,
                                        const int num_outliers = 2);

} // namespace limap
