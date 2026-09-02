#pragma once

#include <optional>

#include "limap/geometry/line2d.h"
#include "limap/util/eigen_types.h"

namespace limap {

// InfiniteLine2d with homogeneous line coordinate
class InfiniteLine2d {
public:
  // Default: leaves coords uninitialized.
  InfiniteLine2d() = default;

  // Construct from homogeneous line coordinates (normalized internally).
  explicit InfiniteLine2d(const V3D &coords_);

  // Construct from a point on the line and a unit direction vector.
  InfiniteLine2d(const V2D &p, const V2D &direc);

  // Construct from an existing Line2d.
  explicit InfiniteLine2d(const Line2d &line);

  // Orthogonal projection of point q onto the line.
  V2D PointProjection(const V2D &q) const;

  // Euclidean distance from point q to the line.
  double PointDistance(const V2D &q) const;

  // Return one concrete point on the line (projection of the origin).
  V2D Point() const;

  // Return a unit direction vector of the line.
  V2D Direction() const;

  // Data: normalized homogeneous line coordinates (a, b, c), with
  // sqrt(a^2+b^2)=1.
  V3D coords;
};

// Intersection of two infinite lines. Returns nullopt if fails.
std::optional<V2D> IntersectInfiniteLine2d(const InfiniteLine2d &l1,
                                           const InfiniteLine2d &l2);

} // namespace limap
