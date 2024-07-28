#ifndef LIMAP_BASE_LINEBASE_H_
#define LIMAP_BASE_LINEBASE_H_

#include <cmath>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <set>

namespace py = pybind11;

#include "base/camera_view.h"
#include "util/types.h"

namespace limap {

class Line2d {
public:
  Line2d() {}
  Line2d(const Eigen::MatrixXd &seg2d);
  Line2d(V2D start, V2D end, double score = -1);
  V2D start, end;
  double score = -1;

  double length() const { return (start - end).norm(); }
  V2D midpoint() const { return 0.5 * (start + end); }
  V2D direction() const { return (end - start).normalized(); }
  V2D perp_direction() const {
    V2D dir = direction();
    return V2D(dir[1], -dir[0]);
  }
  V3D coords() const; // get homogeneous coordinate
  V2D point_projection(const V2D &p) const;
  double point_distance(const V2D &p) const;
  Eigen::MatrixXd as_array() const;
};

class Line3d {
public:
  Line3d() {}
  Line3d(const Eigen::MatrixXd &seg3d);
  Line3d(V3D start, V3D end, double score = -1, double depth_start = -1,
         double depth_end = -1, double uncertainty = -1);
  V3D start, end;
  double score = -1;
  double uncertainty = -1.0;
  V2D depths; // [depth_start, depth_end] for the source perspective image

  void set_uncertainty(const double val) { uncertainty = val; }
  double length() const { return (start - end).norm(); }
  V3D midpoint() const { return 0.5 * (start + end); }
  V3D direction() const { return (end - start).normalized(); }
  V3D point_projection(const V3D &p) const;
  double point_distance(const V3D &p) const;
  Eigen::MatrixXd as_array() const;
  Line2d projection(const CameraView &view) const;
  double sensitivity(const CameraView &view)
      const; // in angle, 0 for perfect view, 90 for collapsing
  double computeUncertainty(const CameraView &view,
                            const double var2d = 5.0) const;
};

std::vector<Line2d> GetLine2dVectorFromArray(const Eigen::MatrixXd &segs2d);
std::vector<Line3d>
GetLine3dVectorFromArray(const std::vector<Eigen::MatrixXd> &segs3d);

Line2d projection_line3d(const Line3d &line3d, const CameraView &view);
Line3d unprojection_line2d(const Line2d &line2d, const CameraView &view,
                           const std::pair<double, double> &depths);

} // namespace limap

#endif
