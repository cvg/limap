#include "limap/geometry/line2d.h"

#include <cmath>
#include <colmap/util/logging.h>

namespace limap {

Line2d::Line2d(V2D start_, V2D end_) {
  start = start_;
  end = end_;
}

Line2d::Line2d(const Eigen::MatrixXd &seg) {
  THROW_CHECK_EQ(seg.rows(), 2);
  THROW_CHECK_EQ(seg.cols(), 2);
  start = V2D(seg(0, 0), seg(0, 1));
  end = V2D(seg(1, 0), seg(1, 1));
}

double Line2d::Length() const {
  // Compute Euclidean length between endpoints
  return (start - end).norm();
}

V2D Line2d::Midpoint() const {
  // Average of endpoints
  return 0.5 * (start + end);
}

V2D Line2d::Direction() const {
  // Normalized direction vector from start to end
  return (end - start).normalized();
}

V2D Line2d::PerpDirection() const {
  // Perpendicular direction (+90 degrees rotation)
  V2D dir = Direction();
  return V2D(dir[1], -dir[0]);
}

V2D Line2d::PointProjection(const V2D &p) const {
  // Project point onto the supporting line, then clamp to the segment
  const V2D dir = Direction();
  const double proj = (p - start).dot(dir);

  if (proj <= 0.0) {
    return start; // before start
  }

  const double len = Length();
  if (proj >= len) {
    return end; // beyond end
  }

  return start + proj * dir; // between start and end
}

double Line2d::PointDistance(const V2D &p) const {
  // Compute Euclidean distance between point and its projection
  const V2D p_proj = PointProjection(p);
  return (p - p_proj).norm();
}

V3D Line2d::Coords() const {
  // Convert endpoints to homogeneous coordinates and compute cross product
  // Gives normalized line equation coefficients (a,b,c)
  const V3D start_h = start.homogeneous();
  const V3D end_h = end.homogeneous();
  return start_h.cross(end_h).normalized();
}

Eigen::MatrixXd Line2d::AsArray() const {
  // Return endpoints as 2x2 array
  Eigen::MatrixXd arr(2, 2);
  arr(0, 0) = start[0];
  arr(0, 1) = start[1];
  arr(1, 0) = end[0];
  arr(1, 1) = end[1];
  return arr;
}

std::vector<Line2d> GetLine2dVectorFromArray(const Eigen::MatrixXd &segs2d) {
  if (segs2d.rows() != 0)
    THROW_CHECK_GE(segs2d.cols(), 4);
  std::vector<Line2d> lines;
  for (int i = 0; i < segs2d.rows(); ++i)
    lines.push_back(Line2d(V2D(segs2d(i, 0), segs2d(i, 1)),
                           V2D(segs2d(i, 2), segs2d(i, 3))));
  return lines;
}

} // namespace limap
