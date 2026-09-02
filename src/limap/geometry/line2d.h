#pragma once

#include "limap/util/eigen_types.h"
#include "limap/util/types.h"

namespace limap {

class Line2d {
public:
  // Default constructor (creates an uninitialized line)
  Line2d() = default;

  // Construct from a 2x2 matrix: row 0 = start (x,y), row 1 = end (x,y)
  explicit Line2d(const Eigen::MatrixXd &seg2d);

  // Construct from two endpoints
  Line2d(V2D start, V2D end);

  // Left (first) endpoint of the line segment
  V2D start;

  // Right (second) endpoint of the line segment
  V2D end;

  // Optional score associated with this segment (default: -1)
  double score = -1.0;

  // ID of the corresponding 3D line (if any)
  line3D_t line3D_id = kInvalidLine3dId;

  // @brief Check if this 2D line has a corresponding 3D line
  bool HasLine3D() const { return line3D_id != kInvalidLine3dId; }

  // @brief Euclidean length of the segment
  double Length() const;

  // @brief Midpoint of the segment
  V2D Midpoint() const;

  // @brief Unit direction vector from start to end
  V2D Direction() const;

  // @brief Unit perpendicular direction (+90° rotation)
  V2D PerpDirection() const;

  // @brief Homogeneous line coordinates (a, b, c) satisfying ax + by + c = 0
  V3D Coords() const;

  /**
   * @brief Orthogonal projection of a point onto the segment.
   *
   * Projects the given point p onto the infinite line, then clamps to
   * the segment endpoints.
   * @param p Query point.
   * @return Closest point on the segment to p.
   */
  V2D PointProjection(const V2D &p) const;

  /**
   * @brief Euclidean distance from a point to the segment.
   * @param p Query point.
   * @return Distance between p and its projection on the segment.
   */
  double PointDistance(const V2D &p) const;

  // @brief Represent the segment as a 2×2 matrix [start; end]
  Eigen::MatrixXd AsArray() const;
};

std::vector<Line2d> GetLine2dVectorFromArray(const Eigen::MatrixXd &segs2d);

} // namespace limap
