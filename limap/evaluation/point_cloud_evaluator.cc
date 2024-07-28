#include "evaluation/point_cloud_evaluator.h"

#include <iostream>
#include <numeric>
#include <queue>
#include <third-party/progressbar.hpp>

namespace limap {

namespace evaluation {

double PointCloudEvaluator::ComputeDistPoint(const V3D &point) {
  V3D p = tree_.query_nearest(point);
  return (point - p).norm();
}

std::vector<double> PointCloudEvaluator::ComputeDistsforEachPoint(
    const std::vector<Line3d> &lines) const {
  size_t n_points = tree_.cloud.pts.size();
  std::vector<double> dists(n_points);

  progressbar bar(n_points);
#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    bar.update();
    V3D p = tree_.point(i);
    double min_dist = std::numeric_limits<double>::max();
    for (auto it = lines.begin(); it != lines.end(); ++it) {
      double dist = it->point_distance(p);
      if (dist < min_dist)
        min_dist = dist;
    }
    dists[i] = min_dist;
  }
  return dists;
}

std::vector<double> PointCloudEvaluator::ComputeDistsforEachPoint_KDTree(
    const std::vector<Line3d> &lines) const {
  size_t n_points = tree_.cloud.pts.size();
  std::vector<double> dists(n_points);

  // sample points uniformly on all the lines and build a kd tree
  // TODO: sample by length
  const int n_samples = 1000;
  std::vector<V3D> line_points;
  std::vector<int> labels;
  for (size_t line_id = 0; line_id < lines.size(); ++line_id) {
    auto &line = lines[line_id];
    double interval = line.length() / (n_samples - 1);
    for (size_t i = 0; i < n_samples; ++i) {
      V3D p = line.start + i * interval * (line.end - line.start);
      line_points.push_back(p);
      labels.push_back(line_id);
    }
  }
  KDTree line_tree;
  line_tree.initialize(line_points);

  progressbar bar(n_points);
#pragma omp parallel for
  for (size_t i = 0; i < n_points; ++i) {
    bar.update();
    V3D p = tree_.point(i);
    std::vector<int> res;
    line_tree.query_knn(p, res, 1);
    int index = res[0];
    int line_id = labels[index];
    double min_dist = lines[line_id].point_distance(p);
    dists[i] = min_dist;
  }
  return dists;
}

} // namespace evaluation

} // namespace limap
