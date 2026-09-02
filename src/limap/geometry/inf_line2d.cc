#include "limap/geometry/inf_line2d.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {

InfiniteLine2d::InfiniteLine2d(const V3D &coords_)
    : coords(coords_.normalized()) {}

InfiniteLine2d::InfiniteLine2d(const V2D &p, const V2D &direc) {
  // Expect a unit direction.
  THROW_CHECK_LT(std::abs(direc.norm() - 1.0), 1e-8);

  // For a unit direction d = (dx, dy), a compatible normalized line is:
  // (a, b, c) = (dy, -dx, -dy*px + dx*py), then normalized.
  V3D coor;
  coor[0] = direc[1];
  coor[1] = -direc[0];
  coor[2] = -direc[1] * p[0] + direc[0] * p[1];
  coords = coor.normalized();
}

InfiniteLine2d::InfiniteLine2d(const Line2d &line) {
  CHECK_GT(line.Length(), 0.0);
  coords = line.Coords(); // assumes already normalized
}

V2D InfiniteLine2d::PointProjection(const V2D &q) const {
  // A perpendicular line through q has direction (dy, -dx) where (dx, dy) is
  // our direction.
  V2D direc = Direction();
  InfiniteLine2d perp(q, V2D(direc[1], -direc[0]));

  // Intersection of two lines given by homogeneous coords is their cross
  // product.
  V3D p_homo = coords.cross(perp.coords);

  // Expect a proper finite point (non-zero w).
  THROW_CHECK_NE(p_homo(2), 0);
  return p_homo.hnormalized();
}

double InfiniteLine2d::PointDistance(const V2D &q) const {
  return (q - PointProjection(q)).norm();
}

V2D InfiniteLine2d::Point() const {
  // Any point; we choose projection of the origin for determinism.
  return PointProjection(V2D(0.0, 0.0));
}

V2D InfiniteLine2d::Direction() const {
  // For line (a, b, c), a unit direction perpendicular to the normal (a, b) is
  // (b, -a). Normalize for safety.
  return V2D(coords[1], -coords[0]).normalized();
}

std::optional<V2D> IntersectInfiniteLine2d(const InfiniteLine2d &l1,
                                           const InfiniteLine2d &l2) {
  const V3D &c1 = l1.coords;
  const V3D &c2 = l2.coords;

  V3D p_homo = c1.cross(c2).normalized();

  // If w ~ 0, intersection is at infinity (parallel lines).
  if (std::abs(p_homo(2)) == 0) {
    return std::nullopt;
  }

  return p_homo.hnormalized();
}

} // namespace limap
