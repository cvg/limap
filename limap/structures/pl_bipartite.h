#ifndef LIMAP_STRUCTURES_PL_BIPARTITE_H
#define LIMAP_STRUCTURES_PL_BIPARTITE_H

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "base/linebase.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"
#include "structures/pl_bipartite_base.h"
#include "util/types.h"

namespace limap {

namespace structures {

typedef Junction<Point2d> Junction2d;
typedef Junction<PointTrack> Junction3d;

struct PL_Bipartite2dConfig {
  PL_Bipartite2dConfig() {}
  PL_Bipartite2dConfig(py::dict dict) {
    ASSIGN_PYDICT_ITEM(dict, threshold_intersection, double)
    ASSIGN_PYDICT_ITEM(dict, threshold_merge_junctions, double)
    ASSIGN_PYDICT_ITEM(dict, threshold_keypoints, double)
  }

  double threshold_keypoints = 2.0;       // in pixels
  double threshold_intersection = 2.0;    // in pixels
  double threshold_merge_junctions = 2.0; // in pixels
};

class PL_Bipartite2d : public PL_Bipartite<Point2d, Line2d> {
public:
  PL_Bipartite2d() {}
  ~PL_Bipartite2d() {}
  PL_Bipartite2d(const PL_Bipartite2dConfig &config) : config_(config) {}
  PL_Bipartite2d(const PL_Bipartite2d &obj)
      : PL_Bipartite<Point2d, Line2d>(obj) {}
  PL_Bipartite2d(py::dict dict);
  py::dict as_dict() const;

  void
  add_keypoint(const Point2d &p,
               int point_id = -1); // compute connection by point-line distance;
  void add_keypoints_with_point3D_ids(
      const std::vector<V2D> &points, const std::vector<int> &point3D_ids,
      const std::vector<int> &ids = std::vector<int>());
  void compute_intersection(); // compute intersections
  void compute_intersection_with_points(
      const std::vector<V2D> &
          points); // compute intersection and remove overlaps with input points

private:
  PL_Bipartite2dConfig config_;
  std::pair<bool, V2D> intersect(const Line2d &l1, const Line2d &l2) const;
  Junction<V2D> merge_junctions(const std::vector<Junction<V2D>> juncs) const;
};

class PL_Bipartite3d : public PL_Bipartite<PointTrack, LineTrack> {
public:
  PL_Bipartite3d() {}
  ~PL_Bipartite3d() {}
  PL_Bipartite3d(const PL_Bipartite3d &obj)
      : PL_Bipartite<PointTrack, LineTrack>(obj) {}
  PL_Bipartite3d(py::dict dict);
  py::dict as_dict() const;

  std::vector<V3D> get_point_cloud() const;
  std::vector<Line3d> get_line_cloud() const;
};

} // namespace structures

} // namespace limap

#endif
