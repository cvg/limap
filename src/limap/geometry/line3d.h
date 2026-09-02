#pragma once

#include <cmath>
#include <set>

#include <colmap/scene/image.h>
#include <colmap/scene/track.h>

#include "limap/geometry/line2d.h"
#include "limap/util/eigen_types.h"
#include "limap/util/types.h"

namespace limap {

class Line3d {
public:
  // Default constructor
  Line3d() = default;

  // Construct from a 2x3 matrix: row 0 = start (x,y,z), row 1 = end (x,y,z)
  explicit Line3d(const Eigen::MatrixXd &seg3d);

  // Construct from endpoints + optional uncertainty
  Line3d(V3D start, V3D end, double uncertainty = -1.0);

  // Endpoints of the 3D segment
  V3D start, end;

  // Associated feature track (from COLMAP)
  colmap::Track track;

  // Optional per-line uncertainty (default: -1)
  double uncertainty = -1.0;

  // Set the uncertainty value
  void SetUncertainty(double val);

  // Euclidean length of the segment
  double Length() const;

  // Midpoint of the segment
  V3D Midpoint() const;

  // Unit direction vector from start to end
  V3D Direction() const;

  // Orthogonal projection of a 3D point onto the segment
  V3D PointProjection(const V3D &p) const;

  // Euclidean distance from a 3D point to the segment
  double PointDistance(const V3D &p) const;

  // Represent as a 2×3 matrix [start; end]
  Eigen::MatrixXd AsArray() const;

  // Project this 3D segment into the given camera view.
  // Return std::nullopt if both endpoints are behind the image.
  std::optional<Line2d> Projection(const colmap::Image &image) const;
  // Projection without cheirality check.
  Line2d AlgProjection(const colmap::Image &image) const;

  /**
   * @brief Sensitivity of this segment in a given view (degrees).
   * 0°   → best (line direction ⟂ viewing ray),
   * 90°  → worst / collapsing (line direction ∥ viewing ray).
   */
  double Sensitivity(const colmap::Image &image) const;

  /**
   * @brief Estimate uncertainty based on depth and 2D variance model.
   * @param view  Camera view containing pose/camera model.
   * @param var2d Variance of 2D detection (default 5.0).
   */
  double ComputeUncertainty(const colmap::Image &image,
                            double var2d = 5.0) const;
};

std::vector<Line3d>
GetLine3dVectorFromArray(const std::vector<Eigen::MatrixXd> &segs3d);

// Return std::nullopt if both endpoints are behind the image.
std::optional<Line2d> ProjectionLine3d(const Line3d &line3d,
                                       const colmap::Image &image);

Line3d UnprojectionLine2d(const Line2d &line2d, const colmap::Image &image,
                          const std::pair<double, double> &depths);

Line3d AggregateLine3dList(const std::vector<Line3d> &lines,
                           int num_outliers = 2);

} // namespace limap
