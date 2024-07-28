#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#include "util/kd_tree.h"
#include "util/nanoflann.hpp"

namespace limap {

using namespace nanoflann;

bool KDTree::empty() const { return cloud.pts.empty(); }

void KDTree::initialize(const Eigen::MatrixXd &points, bool build_index) {
  std::vector<Eigen::Vector3d> points_tmp;
  for (int i = 0; i < points.rows(); ++i) {
    points_tmp.push_back(points.row(i));
  }
  initialize(points_tmp, build_index);
}

void KDTree::initialize(const std::vector<Eigen::Vector3d> &points,
                        bool build_index) {
  if (points.empty())
    return;
  cloud.pts = points;
  index = std::unique_ptr<my_kd_tree_t>(
      new my_kd_tree_t(3, cloud, KDTreeSingleIndexAdaptorParams(10)));
  if (build_index)
    buildIndex();
}

void KDTree::buildIndex() {
  index->buildIndex();
  if (options_.do_print)
    std::cout << "KD-tree initialized (" << cloud.pts.size() << " points)"
              << std::endl;
}

Eigen::Vector3d KDTree::query_nearest(const Eigen::Vector3d &query_pt) const {
  std::vector<int> indexes;
  query_knn(query_pt, indexes, 1);
  return point(indexes[0]);
}

void KDTree::query_knn(const Eigen::Vector3d &query_pt,
                       std::vector<int> &nearestVerticesIdx,
                       size_t num_results) const {
  // ----------------------------------------------------------------
  // knnSearch():  Perform a search for the N closest points
  // ----------------------------------------------------------------
  std::vector<uint32_t> ret_index(num_results);
  std::vector<double> out_dist_sqr(num_results);

  double query_pt_tmp[3] = {query_pt(0), query_pt(1), query_pt(2)};
  num_results = index->knnSearch(&query_pt_tmp[0], num_results, &ret_index[0],
                                 &out_dist_sqr[0]);

  // In case of less points in the tree than requested:
  ret_index.resize(num_results);
  out_dist_sqr.resize(num_results);

  // construct nearest vertices
  nearestVerticesIdx.clear();
  for (size_t i = 0; i < num_results; i++)
    nearestVerticesIdx.push_back(ret_index[i]);

  // printing results
  if (options_.do_print) {
    std::cout << "knnSearch(): num_results=" << num_results << "\n";
    for (size_t i = 0; i < num_results; i++)
      std::cout << "idx[" << i << "]=" << ret_index[i] << " dist[" << i
                << "]=" << out_dist_sqr[i] << std::endl;
    std::cout << "\n";
  }
}

void KDTree::query_radius_search(const Eigen::Vector3d &query_pt,
                                 std::vector<int> &nearestVerticesIdx,
                                 double search_radius) const {
  // ----------------------------------------------------------------
  // radiusSearch(): Perform a search for the points within search_radius
  // ----------------------------------------------------------------
  // const double search_radius = static_cast<double>(0.1);

  // radius search
  std::vector<std::pair<uint32_t, double>> ret_matches;
  nanoflann::SearchParams params;
  double query_pt_tmp[3] = {query_pt[0], query_pt[1], query_pt[2]};
  const size_t nMatches = index->radiusSearch(
      &query_pt_tmp[0], search_radius * search_radius, ret_matches, params);

  // get nearest vertices index
  nearestVerticesIdx.clear();
  for (size_t i = 0; i < nMatches; i++)
    nearestVerticesIdx.push_back(ret_matches[i].first);

  // printing results
  if (options_.do_print) {
    std::cout << "radiusSearch(): radius=" << search_radius << " -> "
              << nMatches << " matches\n";
    for (size_t i = 0; i < nMatches; i++)
      std::cout << "idx[" << i << "]=" << ret_matches[i].first << " dist[" << i
                << "]=" << ret_matches[i].second << std::endl;
    std::cout << std::endl;
  }
}

void KDTree::save(const std::string &filename) const {
  std::ofstream f(filename, std::ofstream::binary);
  if (f.bad())
    throw std::runtime_error("Error writing index file!");
  index->saveIndex(f);
  f.close();
}

void KDTree::load(const std::string &filename) {
  std::ifstream f(filename, std::ifstream::binary);
  if (f.fail())
    throw std::runtime_error("Error reading index file!");
  index->loadIndex(f);
  f.close();
  std::cout << "KD-tree loaded (" << cloud.pts.size() << " points)"
            << std::endl;
}

} // namespace limap
