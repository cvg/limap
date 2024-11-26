#ifndef LIMAP_BASE_INFINITE_LINE_H_
#define LIMAP_BASE_INFINITE_LINE_H_

#include "_limap/helpers.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "base/camera_view.h"
#include "base/linebase.h"
#include "util/types.h"

#include <ceres/ceres.h>

namespace py = pybind11;

namespace limap {

// InfiniteLine2d with homogeneous line coordinate
class InfiniteLine2d {
public:
  InfiniteLine2d() {}
  InfiniteLine2d(const V3D &coords_) : coords(coords_.normalized()) {}
  InfiniteLine2d(const V2D &p, const V2D &direc);
  InfiniteLine2d(const Line2d &line);

  V2D point_projection(const V2D &q) const;
  double point_distance(const V2D &q) const;

  V2D point() const;     // get a point on the line
  V2D direction() const; // get the direction of the line

  // data
  V3D coords; // homogeneous line coordinate
};

std::pair<V2D, bool> Intersect_InfiniteLine2d(const InfiniteLine2d &l1,
                                              const InfiniteLine2d &l2);

// InfiniteLine3d with Plucker coordinate
class InfiniteLine3d {
public:
  InfiniteLine3d() {}
  // use_normal == True -> (a, b) is (p, direc): normal coordinate with a point
  // and a direction use_normal == False -> (a, b) is (direc, m): plucker
  // coordinate
  InfiniteLine3d(const V3D &a, const V3D &b, bool use_normal = true);
  InfiniteLine3d(const Line3d &line);

  V3D point_projection(const V3D &q) const;
  double point_distance(const V3D &q) const;
  InfiniteLine2d projection(const CameraView &view) const;
  V3D unprojection(const V2D &p2d, const CameraView &view) const;
  V3D project_from_infinite_line(const InfiniteLine3d &line) const;
  V3D project_to_infinite_line(const InfiniteLine3d &line) const;

  V3D point() const;     // get a point on the line
  V3D direction() const; // get the direction of the line
  M4D matrix() const;    // get Plucker matrix. [LINK]
                         // https://en.wikipedia.org/wiki/Pl%C3%BCcker_matrix

  // data
  V3D d; // direction
  V3D m; // moment
};

// minimal Plucker coordinate used for ceres optimization
class MinimalInfiniteLine3d {
public:
  MinimalInfiniteLine3d() {}
  MinimalInfiniteLine3d(const Line3d &line)
      : MinimalInfiniteLine3d(InfiniteLine3d(line)) {};
  MinimalInfiniteLine3d(const InfiniteLine3d &inf_line);
  MinimalInfiniteLine3d(const std::vector<double> &values);
  InfiniteLine3d GetInfiniteLine() const;

  V4D uvec; // quaternion vector for SO(3)
  V2D wvec; // homogenous vector for SO(2)
};

Line3d GetLineSegmentFromInfiniteLine3d(
    const InfiniteLine3d &inf_line, const std::vector<CameraView> &views,
    const std::vector<Line2d> &line2ds,
    const int num_outliers = 2); // views.size() == line2ds.size()
Line3d GetLineSegmentFromInfiniteLine3d(const InfiniteLine3d &inf_line,
                                        const std::vector<Line3d> &line3ds,
                                        const int num_outliers = 2);

} // namespace limap

#endif
