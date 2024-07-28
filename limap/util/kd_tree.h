#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string.h>

#include "util/nanoflann.hpp"
#include <Eigen/Core>

using namespace nanoflann;

namespace limap {

template <typename T> struct PointCloud {
  std::vector<Eigen::Vector3d> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    // if (dim == 0) return pts[idx].x;
    // else if (dim == 1) return pts[idx].y;
    // else return pts[idx].z;
    if (dim == 0)
      return pts[idx](0);
    else if (dim == 1)
      return pts[idx](1);
    else
      return pts[idx](2);
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

struct KDTreeOptions {
  bool do_print = false;
};

class KDTree {
public:
  // Originally an empty tree pointer here.
  // Filled with function initialize(const Eigen::MatrixXd &points)
  static std::vector<int> DEFAULT_VECTOR;
  typedef KDTreeSingleIndexAdaptor<
      L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>,
      3 /* dim */
      >
      my_kd_tree_t;
  std::unique_ptr<my_kd_tree_t> index = NULL;

  KDTreeOptions options_;
  PointCloud<double> cloud;

  KDTree() {}
  ~KDTree() {}
  KDTree(const KDTreeOptions &options) : options_(options) {}
  KDTree(const Eigen::MatrixXd &points) { initialize(points); }
  KDTree(const KDTreeOptions &options, const Eigen::MatrixXd &points)
      : options_(options) {
    initialize(points);
  }
  KDTree(const std::vector<Eigen::Vector3d> &points) { initialize(points); }
  KDTree(const KDTreeOptions &options,
         const std::vector<Eigen::Vector3d> &points)
      : options_(options) {
    initialize(points);
  }

  bool empty() const;
  std::vector<Eigen::Vector3d> all_points() { return cloud.pts; }
  Eigen::Vector3d point(const int &id) const {
    if (id >= cloud.pts.size()) {
      throw std::runtime_error("query id is out of range");
    } else {
      return cloud.pts[id];
    }
  }

  void initialize(const std::vector<Eigen::Vector3d> &points,
                  bool build_index = true);
  void initialize(const Eigen::MatrixXd &points, bool build_index = true);
  void buildIndex();
  Eigen::Vector3d query_nearest(const Eigen::Vector3d &query_pt) const;
  double point_distance(const Eigen::Vector3d &query_pt) const {
    return (query_pt - query_nearest(query_pt)).norm();
  }
  void query_knn(const Eigen::Vector3d &query_pt,
                 std::vector<int> &nearestVerticesIdx,
                 size_t num_results = 5) const;
  void query_radius_search(const Eigen::Vector3d &query_pt,
                           std::vector<int> &nearestVerticesIdx,
                           double search_radius = 0.62) const;

  void save(const std::string &filename) const;
  void load(const std::string &filename);
};

} // namespace limap
